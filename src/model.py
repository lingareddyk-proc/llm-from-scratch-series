"""A small modern Llama/Gemma-style decoder LLM, used from Post 7 onwards.

Design goals:
- Educational: every block is one short, readable class.
- Modern: RMSNorm + RoPE + Grouped-Query Attention + SwiGLU + Pre-Norm residuals.
- Small enough to pretrain on free Colab T4 (default config ~10M params).
- No exotic engineering tricks (no FlashAttention call, no fused kernels) so the
  math stays visible.

Used by:
- Post 7  — assembles the block and shows the forward pass shape-by-shape.
- Post 8  — pretrains the full model on TinyStories.
- Post 9  — adds a KV-cache for fast inference.
- Post 10 — swaps the FFN for an MoE layer.
- Post 11 — fine-tunes (SFT + DPO).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 8_000
    d_model: int = 256
    n_layers: int = 6
    n_heads_q: int = 8        # number of query heads
    n_heads_kv: int = 2       # number of K/V heads (GQA: n_heads_q must be a multiple)
    d_ff: int = 768           # SwiGLU intermediate size
    max_seq_len: int = 512
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads_q == 0
        return self.d_model // self.n_heads_q

    @property
    def n_kv_groups(self) -> int:
        assert self.n_heads_q % self.n_heads_kv == 0
        return self.n_heads_q // self.n_heads_kv


# ---------------------------------------------------------------------------
# RMSNorm  (Llama / Gemma normalisation; no mean-centering, no bias)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def build_rope_cache(d_head: int, max_seq_len: int, base: float = 10000.0):
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., seq, d_head)   cos/sin: (max_seq_len, d_head/2)"""
    seq = x.size(-2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[:seq]
    s = sin[:seq]
    rot1 = x1 * c - x2 * s
    rot2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ---------------------------------------------------------------------------
# Grouped-Query Attention with RoPE and causal mask
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads_q * cfg.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_o = nn.Linear(cfg.n_heads_q * cfg.d_head, cfg.d_model, bias=False)
        cos, sin = build_rope_cache(cfg.d_head, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Hq, Hkv, Dh = self.cfg.n_heads_q, self.cfg.n_heads_kv, self.cfg.d_head

        q = self.W_q(x).view(B, T, Hq, Dh).transpose(1, 2)   # (B, Hq, T, Dh)
        k = self.W_k(x).view(B, T, Hkv, Dh).transpose(1, 2)  # (B, Hkv, T, Dh)
        v = self.W_v(x).view(B, T, Hkv, Dh).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # GQA: repeat K/V heads to match Q heads
        if Hkv != Hq:
            repeat = Hq // Hkv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = q @ k.transpose(-1, -2) / (Dh ** 0.5)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, Hq * Dh)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Decoder block: Pre-Norm + GQA + Pre-Norm + SwiGLU
# ---------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full decoder LLM
# ---------------------------------------------------------------------------


class TinyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head_weight = self.tok_emb.weight  # tied
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        if self.cfg.tie_embeddings:
            return x @ self.lm_head_weight.T
        return self.lm_head(x)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            # tok_emb.weight is double-counted in tied configuration since lm_head_weight aliases it
            pass
        return n
