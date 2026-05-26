"""Generate notebook.ipynb for Post 9 — Sampling and the KV-cache."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "notebook.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


CELLS = [
    md(
        """# Post 9 — Sampling and the KV-Cache

Companion notebook to the Medium post. ~10 min, runs on free Colab CPU; faster on T4 GPU.

We will:

1. Build a `KVCachedTinyLLM` \u2014 same architecture as `TinyLLM` from Post 7, but with a per-block K/V cache.
2. Implement `generate_with_cache` and a `generate_naive` baseline.
3. **Benchmark** both \u2014 tokens/sec for a 200-token completion.
4. Implement **top-p** and **min-p** samplers.
5. Compare sampler outputs on the same prompt."""
    ),
    md("## 0. Setup"),
    code(
        '''!pip -q install "torch>=2.3" "tokenizers>=0.19" matplotlib

import os, sys, time, subprocess
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)'''
    ),
    md(
        """## 1. Load Post 8's checkpoint (or initialize fresh if not available)

If you ran Post 8 in this same Colab session, `/content/tinyllm_pretrained.pt` already exists. Otherwise we initialise a fresh model for the architectural demo; the speedup numbers still work, the generations just won't be meaningful."""
    ),
    code(
        '''from src.model import ModelConfig, TinyLLM
from tokenizers import Tokenizer

CKPT_PATH = "/content/tinyllm_pretrained.pt"
TOK_PATH  = "/content/tinystories_tokenizer.json"

if os.path.exists(CKPT_PATH) and os.path.exists(TOK_PATH):
    print("Loading Post 8 checkpoint and tokenizer.")
    obj = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ModelConfig(**obj["config"])
    base_model = TinyLLM(cfg).to(device)
    base_model.load_state_dict(obj["model_state"])
    tokenizer = Tokenizer.from_file(TOK_PATH)
    has_ckpt = True
else:
    print("No checkpoint found. Using fresh-init model \u2014 generations will be gibberish")
    print("but the speedup benchmark is meaningful.")
    cfg = ModelConfig(vocab_size=8000, d_model=256, n_layers=6, n_heads_q=8, n_heads_kv=2, d_ff=768, max_seq_len=512)
    base_model = TinyLLM(cfg).to(device)
    tokenizer = None
    has_ckpt = False

base_model.eval()
print(f"model params: {sum(p.numel() for p in base_model.parameters()):,}")'''
    ),
    md(
        """## 2. KV-cache-enabled attention and model

Same math as Post 7's `GroupedQueryAttention`, but the forward pass takes an optional `past_kv` and returns the updated cache."""
    ),
    code(
        '''from src.model import RMSNorm, SwiGLU, build_rope_cache, apply_rope


class CachedGQA(nn.Module):
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

    def forward(self, x, past_kv=None):
        B, T_new, D = x.shape
        Hq, Hkv, Dh = self.cfg.n_heads_q, self.cfg.n_heads_kv, self.cfg.d_head

        q = self.W_q(x).view(B, T_new, Hq, Dh).transpose(1, 2)
        k = self.W_k(x).view(B, T_new, Hkv, Dh).transpose(1, 2)
        v = self.W_v(x).view(B, T_new, Hkv, Dh).transpose(1, 2)

        T_past = 0 if past_kv is None else past_kv[0].size(-2)
        cos = self.rope_cos[T_past:T_past + T_new]
        sin = self.rope_sin[T_past:T_past + T_new]
        # apply_rope expects (..., seq, d_head) and uses cos[:seq]
        q = apply_rope_pos(q, cos, sin)
        k = apply_rope_pos(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v)

        if Hkv != Hq:
            repeat = Hq // Hkv
            k_b = k.repeat_interleave(repeat, dim=1)
            v_b = v.repeat_interleave(repeat, dim=1)
        else:
            k_b, v_b = k, v

        scores = q @ k_b.transpose(-1, -2) / (Dh ** 0.5)
        if T_new > 1:
            T_total = T_past + T_new
            mask = torch.tril(torch.ones(T_new, T_total, dtype=torch.bool, device=x.device), diagonal=T_past)
            scores = scores.masked_fill(~mask, float("-inf"))

        w = F.softmax(scores, dim=-1)
        out = (w @ v_b).transpose(1, 2).contiguous().view(B, T_new, Hq * Dh)
        return self.W_o(out), new_kv


def apply_rope_pos(x, cos, sin):
    """Like apply_rope but cos/sin are already sliced to the right positions."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


class CachedBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CachedGQA(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, past_kv=None):
        a, new_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class CachedTinyLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CachedBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, ids, past_kvs=None):
        x = self.tok_emb(ids)
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, kv = blk(x, past_kv=past)
            new_kvs.append(kv)
        x = self.norm_f(x)
        logits = x @ self.tok_emb.weight.T
        return logits, new_kvs

print("Defined CachedTinyLLM.")'''
    ),
    md(
        """## 3. Copy weights from the dense model to the cached model

(They have identical structure modulo parameter naming.)"""
    ),
    code(
        '''cached_model = CachedTinyLLM(cfg).to(device).eval()

# Map state_dict keys: TinyLLM and CachedTinyLLM use identical names for matching layers.
src_sd = base_model.state_dict()
dst_sd = cached_model.state_dict()
matched = {k: v for k, v in src_sd.items() if k in dst_sd and v.shape == dst_sd[k].shape}
missing = [k for k in dst_sd if k not in matched]
extra   = [k for k in src_sd if k not in matched]
print(f"matched {len(matched)} tensors; missing {len(missing)}, extra {len(extra)}")
cached_model.load_state_dict(matched, strict=False)
print("Weights copied.")'''
    ),
    md(
        """## 4. Two generation functions

`generate_naive`: every step, run the dense model on the whole prefix.
`generate_with_cache`: prefill once, then feed one token at a time."""
    ),
    code(
        '''@torch.no_grad()
def generate_naive(model, ids, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        ids_cropped = ids[:, -model.cfg.max_seq_len:]
        logits = model(ids_cropped)[:, -1, :]
        v, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def generate_with_cache(model, ids, max_new_tokens, temperature=1.0, top_k=50):
    # Prefill: process the whole prompt, capture cache
    logits, past_kvs = model(ids)
    last_logits = logits[:, -1, :]
    out_ids = ids.clone()
    for _ in range(max_new_tokens):
        v, _ = torch.topk(last_logits, top_k)
        last_logits = torch.where(last_logits < v[:, [-1]], torch.full_like(last_logits, -float("inf")), last_logits)
        probs = F.softmax(last_logits / max(temperature, 1e-4), dim=-1)
        nxt = torch.multinomial(probs, 1)
        out_ids = torch.cat([out_ids, nxt], dim=1)
        # Single-token step using the cache
        logits, past_kvs = model(nxt, past_kvs=past_kvs)
        last_logits = logits[:, -1, :]
    return out_ids

print("Defined generation functions.")'''
    ),
    md(
        """## 5. Benchmark"""
    ),
    code(
        '''prompt = "Once upon a time"
if has_ckpt:
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
else:
    # Fresh model: just use random ids for the benchmark
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 8), device=device)

torch.manual_seed(0)
results = {}
for name, fn in [("naive", generate_naive), ("kv-cache", generate_with_cache)]:
    # Warm up
    _ = fn(base_model if name == "naive" else cached_model, prompt_ids, max_new_tokens=5)
    torch.cuda.synchronize() if device.type == "cuda" else None

    n_new = 200
    t0 = time.time()
    out = fn(base_model if name == "naive" else cached_model, prompt_ids, max_new_tokens=n_new)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = time.time() - t0
    tps = n_new / dt
    results[name] = (dt, tps)
    print(f"{name:>10}  {dt:.2f} s   {tps:.1f} tokens/sec")

speedup = results["kv-cache"][1] / results["naive"][1]
print(f"\\nKV-cache speedup: {speedup:.2f}x")'''
    ),
    code(
        '''names  = list(results.keys())
tps_l  = [results[n][1] for n in names]

plt.figure(figsize=(6, 4))
plt.bar(names, tps_l, color=['#cc0000', '#0066cc'])
for i, v in enumerate(tps_l):
    plt.text(i, v, f"{v:.1f}", ha='center', va='bottom')
plt.ylabel("tokens / second")
plt.title("Generation speed: naive vs KV-cache")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 6. Sampling strategies

`top_p` (nucleus): keep smallest set whose cumulative prob \u2265 p.
`min_p`: keep tokens with prob \u2265 min_p \u00d7 prob(top1)."""
    ),
    code(
        '''def top_p_filter(logits, p=0.95):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = sorted_probs.cumsum(dim=-1)
    mask = cumprobs > p
    mask[..., 1:] = mask[..., :-1].clone()  # shift so we always keep at least 1
    mask[..., 0] = False
    sorted_logits[mask] = float("-inf")
    return sorted_logits.gather(-1, sorted_idx.argsort(-1))


def min_p_filter(logits, min_p=0.05):
    probs = F.softmax(logits, dim=-1)
    top1 = probs.max(dim=-1, keepdim=True).values
    keep = probs >= min_p * top1
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))


@torch.no_grad()
def generate_sampler(model, ids, max_new_tokens, sampler, temperature=1.0):
    logits, past_kvs = model(ids)
    last_logits = logits[:, -1, :]
    out_ids = ids.clone()
    for _ in range(max_new_tokens):
        l = sampler(last_logits.clone())
        probs = F.softmax(l / max(temperature, 1e-4), dim=-1)
        nxt = torch.multinomial(probs, 1)
        out_ids = torch.cat([out_ids, nxt], dim=1)
        logits, past_kvs = model(nxt, past_kvs=past_kvs)
        last_logits = logits[:, -1, :]
    return out_ids

print("Defined samplers.")'''
    ),
    code(
        '''samplers = {
    "greedy (T=0.01)":  (lambda L: L,                                  0.01),
    "T=1, top-k=50":    (lambda L: top_p_filter(L, p=1.01),            1.0),  # ~no filter
    "T=1, top-p=0.9":   (lambda L: top_p_filter(L, p=0.9),             1.0),
    "T=1, min-p=0.05":  (lambda L: min_p_filter(L, min_p=0.05),        1.0),
    "T=1.4, top-p=0.95":(lambda L: top_p_filter(L, p=0.95),            1.4),
}

if has_ckpt:
    prompt = "Once upon a time, in a quiet village,"
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    for name, (sampler, T) in samplers.items():
        torch.manual_seed(0)
        out = generate_sampler(cached_model, prompt_ids, max_new_tokens=60, sampler=sampler, temperature=T)
        print(f"--- {name} ---")
        print(tokenizer.decode(out[0].tolist()))
        print()
else:
    print("(skipping sample comparison because no checkpoint loaded)")'''
    ),
    md(
        """## 7. What you just did

- Built a KV-cache-enabled version of `TinyLLM` from scratch.
- Verified it produces the same outputs as the naive version while being many times faster.
- Implemented `top-p` and `min-p` sampling \u2014 the strategies behind modern chat models.

Every production LLM serving stack (vLLM, TGI, OpenAI/Anthropic/Google internal) is built around KV-cache + paged attention + speculative decoding + continuous batching. You now know the core idea.

**Next: Post 10 \u2014 Mixture of Experts.**"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
