"""Generate notebook.ipynb for Post 6 — Positional Encoding & RoPE."""

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
        """# Post 6 — Positional Encoding & RoPE

Companion notebook to the Medium post. ~5 min on free Colab CPU.

We will:

1. Demonstrate that **attention is permutation-invariant** without positions.
2. Implement **sinusoidal**, **learned**, and **RoPE** positional schemes.
3. **Visualize RoPE rotations** in 2-D.
4. **Numerically verify** that the dot product `Q'(m) \u00b7 K'(n)` after RoPE depends only on `n - m`."""
    ),
    md("## 0. Setup"),
    code('!pip -q install "torch>=2.3" matplotlib numpy'),
    code(
        """import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)"""
    ),
    md(
        """## 1. Attention is permutation-invariant without positions

We feed a small sequence and a permuted version of it through attention. Without positional info, the *set* of output vectors is identical — just in different order."""
    ),
    code(
        '''def causal_attention(X):
    """Single-head, *non*-causal so permutation invariance is obvious."""
    d = X.size(-1)
    Q = K = V = X
    scores = Q @ K.T / math.sqrt(d)
    w = F.softmax(scores, dim=-1)
    return w @ V


X = torch.randn(4, 8)
perm = torch.tensor([2, 0, 3, 1])

out  = causal_attention(X)
out_perm_then_attn = causal_attention(X[perm])

print("Original outputs:")
print(out.round(decimals=2))
print()
print("Permuted-input outputs:")
print(out_perm_then_attn.round(decimals=2))
print()
print("Permuted set matches?", torch.allclose(out[perm], out_perm_then_attn, atol=1e-6))'''
    ),
    md(
        """## 2. Sinusoidal positional embeddings (Vaswani et al. 2017)"""
    ),
    code(
        '''def sinusoidal_pe(seq_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(seq_len).unsqueeze(1)              # (seq, 1)
    i   = torch.arange(d).unsqueeze(0)                    # (1, d)
    div = torch.exp(-(math.log(10000.0) * (i // 2 * 2) / d))
    pe  = torch.zeros(seq_len, d)
    pe[:, 0::2] = torch.sin(pos * div[:, 0::2])
    pe[:, 1::2] = torch.cos(pos * div[:, 1::2])
    return pe

PE = sinusoidal_pe(64, 32)
plt.figure(figsize=(7, 4))
plt.imshow(PE.numpy(), aspect='auto', cmap='RdBu_r')
plt.xlabel("dimension"); plt.ylabel("position")
plt.title("Sinusoidal positional embeddings (Vaswani 2017)")
plt.colorbar(); plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 3. Learned positional embeddings (GPT-2 style)"""
    ),
    code(
        '''class LearnedPE(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)
    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        return x + self.pe(pos)

learned = LearnedPE(1024, 32)
print("learned.pe.weight shape:", learned.pe.weight.shape)
print("That's GPT-2's `wpe` matrix \u2014 1024 rows of length 768 in the real model.")'''
    ),
    md(
        """## 4. RoPE \u2014 Rotary Position Embeddings

The reference implementation: split `d_head` into pairs, rotate each pair by an angle that depends on position **and** which pair you're in. Lower pairs rotate fast (short-range), higher pairs rotate slow (long-range)."""
    ),
    code(
        '''def build_rope_cache(d_head: int, max_seq_len: int, base: float = 10000.0):
    """Returns cos, sin of shape (max_seq_len, d_head/2)."""
    assert d_head % 2 == 0
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))  # (d_head/2,)
    t = torch.arange(max_seq_len).float()                            # (seq,)
    angles = torch.outer(t, freqs)                                   # (seq, d_head/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (..., seq, d_head)   cos/sin: (seq, d_head/2)   returns same shape as x."""
    *prefix, seq, d = x.shape
    x1 = x[..., 0::2]                # even dims
    x2 = x[..., 1::2]                # odd dims
    c = cos[:seq]                    # (seq, d/2)
    s = sin[:seq]
    rot1 = x1 * c - x2 * s
    rot2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


cos, sin = build_rope_cache(d_head=8, max_seq_len=32)
v = torch.randn(32, 8)
v_rot = apply_rope(v, cos, sin)
print("input  shape:", v.shape)
print("output shape:", v_rot.shape)'''
    ),
    md(
        """## 5. Visualize the rotation \u2014 watch a 2-D pair spin as position increases"""
    ),
    code(
        '''cos, sin = build_rope_cache(d_head=8, max_seq_len=16)
v = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])  # constant input, one row
positions = torch.arange(16).unsqueeze(-1).expand(-1, 8).float()  # only for shape

# We rotate the constant vector at each position separately to make the spin obvious.
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), subplot_kw={'aspect': 'equal'})
for pair_idx, ax in enumerate(axes):
    xs, ys = [], []
    for pos in range(16):
        v_pos = apply_rope(v.expand(pos + 1, -1), cos, sin)[pos]  # rotate v at this position
        i, j = 2 * pair_idx, 2 * pair_idx + 1
        xs.append(v_pos[i].item()); ys.append(v_pos[j].item())
    ax.plot(xs, ys, '-o', markersize=4)
    ax.set_title(f"pair {pair_idx} (dims {2*pair_idx},{2*pair_idx+1})")
    ax.grid(alpha=0.3); ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
plt.suptitle("RoPE rotation of a constant input across positions 0..15\\nLow-index pairs spin fast; high-index pairs spin slow")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 6. The key property: dot product after RoPE depends **only on relative position**

For a fixed `q` and `k`, we compute `<RoPE(q, m), RoPE(k, n)>` for many `(m, n)` pairs. We'll plot it as a heatmap and confirm: the value is constant along each *diagonal* (where `n - m` is constant)."""
    ),
    code(
        '''d_head, max_seq_len = 16, 32
cos, sin = build_rope_cache(d_head, max_seq_len)

q = torch.randn(d_head)
k = torch.randn(d_head)

dots = torch.zeros(max_seq_len, max_seq_len)
for m in range(max_seq_len):
    qm = apply_rope(q.unsqueeze(0).expand(m + 1, -1), cos, sin)[m]
    for n in range(max_seq_len):
        kn = apply_rope(k.unsqueeze(0).expand(n + 1, -1), cos, sin)[n]
        dots[m, n] = qm @ kn

plt.figure(figsize=(6, 5))
plt.imshow(dots.numpy(), cmap='RdBu_r', vmin=-dots.abs().max(), vmax=dots.abs().max())
plt.xlabel("key position n")
plt.ylabel("query position m")
plt.title("<RoPE(q, m), RoPE(k, n)>\\n constant along each diagonal -> depends only on n-m")
plt.colorbar(); plt.tight_layout(); plt.show()

# Verify numerically along one diagonal
diag = [dots[i, i + 3].item() for i in range(max_seq_len - 3)]
print(f"diagonal n-m=+3: min={min(diag):.4f}  max={max(diag):.4f}  range={max(diag)-min(diag):.2e}")'''
    ),
    md(
        """## 7. Drop RoPE into our `MultiHeadAttention` from Post 5

A 10-line modification. We rotate Q and K *before* the dot product. V is untouched."""
    ),
    code(
        '''class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, causal: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o   = nn.Linear(d_model, d_model, bias=False)
        cos, sin = build_rope_cache(self.d_head, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        # ----- the only addition relative to Post 5: rotate Q and K -----
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        # ---------------------------------------------------------------

        scores = q @ k.transpose(-1, -2) / (Dh ** 0.5)
        if self.causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

mha = MultiHeadAttentionRoPE(d_model=64, n_heads=8)
x = torch.randn(2, 12, 64)
y = mha(x)
print(f"output shape: {tuple(y.shape)}   (matches Post 5 \u2014 RoPE is a drop-in replacement)")'''
    ),
    md(
        """## 8. What you just did

- Verified that attention without positions is order-blind.
- Implemented sinusoidal, learned, and RoPE positional schemes.
- Visualized RoPE as actual rotations in 2-D, with the frequency varying per pair.
- Proved numerically that **dot products after RoPE depend only on relative position** (constant along diagonals of the dot-product matrix).
- Dropped RoPE into the `MultiHeadAttention` from Post 5 \u2014 the same module we'll use in **Post 7** to build the full Llama/Gemma-style decoder block.

**Next: Post 7 \u2014 The full modern decoder block (Llama / Gemma style).**"""
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
