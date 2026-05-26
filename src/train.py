"""Pretraining utilities used by Post 8.

Kept intentionally small so the whole training loop is readable in one screen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 256
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 100
    total_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 250
    eval_prompt: str = "Once upon a time"
    eval_max_new_tokens: int = 60
    eval_temperature: float = 0.8
    autocast_dtype: torch.dtype = torch.bfloat16


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_max * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def make_batch_sampler(token_ids: torch.Tensor, batch_size: int, seq_len: int, device):
    """Yield random (inputs, targets) batches from a 1-D tensor of token ids."""
    N = token_ids.numel() - seq_len - 1

    def sampler() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            ix = torch.randint(0, N, (batch_size,))
            inputs = torch.stack([token_ids[i : i + seq_len] for i in ix]).to(device)
            targets = torch.stack([token_ids[i + 1 : i + 1 + seq_len] for i in ix]).to(device)
            yield inputs, targets

    return sampler()


def estimate_loss(model: nn.Module, sampler: Iterator, n_batches: int = 10) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = next(sampler)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 100,
    eos_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ids_cropped = ids[:, -model.cfg.max_seq_len :]
        logits = model(ids_cropped)[:, -1, :]
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    model.train()
    return ids


def train(
    model: nn.Module,
    train_sampler: Iterator,
    val_sampler: Iterator,
    cfg: TrainConfig,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> dict:
    """Returns a history dict with train losses and (step, val_loss) pairs."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    device = next(model.parameters()).device

    history = {"train": [], "val": []}
    t0 = time.time()
    model.train()
    for step in range(1, cfg.total_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        x, y = next(train_sampler)
        with torch.autocast(device_type=device.type, dtype=cfg.autocast_dtype, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            tps = step * cfg.batch_size * cfg.seq_len / elapsed
            print(f"step {step:>5}/{cfg.total_steps}  loss {loss.item():.3f}  lr {cosine_lr(step, cfg):.2e}  tokens/s {tps:,.0f}")
            history["train"].append((step, loss.item()))

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            val = estimate_loss(model, val_sampler)
            history["val"].append((step, val))
            print(f"  >> val loss {val:.3f}")
            if on_eval is not None:
                on_eval(step, loss.item(), val)

    return history
