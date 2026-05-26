"""Generate notebook.ipynb for Post 5 — Multi-Head Attention & Specialization."""

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
        """# Post 5 — Multi-Head Attention & Head Specialization

Companion notebook to the Medium post. ~10 min on free Colab CPU.

We will:

1. Implement multi-head attention in vectorized PyTorch (no Python loop over heads).
2. Sanity-check it against a stack of single-head attentions.
3. Load pretrained GPT-2 and **visualize all 144 heads** at once.
4. Programmatically identify a **previous-token head**, a **positional head**, and an **induction head**."""
    ),
    md("## 0. Setup"),
    code('!pip -q install "transformers>=4.44" "torch>=2.3" matplotlib'),
    code(
        """import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
gpt2.eval()
print("Loaded GPT-2 with eager attention.")"""
    ),
    md(
        """## 1. Multi-head attention in vectorized PyTorch

The trick: pack all heads into one tensor of shape `(B, H, T, d_head)`. The math is the same single-head attention you wrote in Post 4 — just with an extra `H` dimension that NumPy / PyTorch broadcasts over."""
    ),
    code(
        '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.W_qkv(x)                                 # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                       # each (B, T, D)
        q = q.view(B, T, H, Dh).transpose(1, 2)              # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / (Dh ** 0.5)     # (B, H, T, T)
        if self.causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                  # (B, H, T, T)

        out = weights @ v                                    # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, D)
        return self.W_o(out), weights

mha = MultiHeadAttention(d_model=64, n_heads=8)
x = torch.randn(2, 5, 64)
out, w = mha(x)
print(f"out shape: {tuple(out.shape)}   weights shape: {tuple(w.shape)}")
print(f"Each (B,H,row) sums to 1? {torch.allclose(w.sum(-1), torch.ones(2, 8, 5), atol=1e-6)}")'''
    ),
    md(
        """## 2. Encode our test sentence with GPT-2 and dump all attention patterns"""
    ),
    code(
        '''sentence = "When John and Mary went to the store, John gave a drink to Mary"
ids = tokenizer.encode(sentence)
tokens = [tokenizer.decode([i]).strip() or "_" for i in ids]
print("tokens:", tokens)

with torch.no_grad():
    out = gpt2(torch.tensor([ids]), output_attentions=True)
attns = torch.stack([a[0] for a in out.attentions])  # (n_layer, n_heads, T, T)
print(f"attention tensor shape: {tuple(attns.shape)}")'''
    ),
    md(
        """## 3. The head zoo: all 144 heads at a glance

12 layers x 12 heads. Eyeball this grid. You'll see strong diagonals (positional / previous-token), vertical bars (heads that attend to a specific token like the BOS), and various more complex patterns."""
    ),
    code(
        '''L, H = attns.shape[:2]
fig, axes = plt.subplots(L, H, figsize=(H * 1.0, L * 1.0))
for li in range(L):
    for hi in range(H):
        ax = axes[li, hi]
        ax.imshow(attns[li, hi].numpy(), cmap='Blues', vmin=0, vmax=attns[li, hi].max().item())
        ax.set_xticks([]); ax.set_yticks([])
        if hi == 0:
            ax.set_ylabel(f"L{li}", fontsize=8)
        if li == 0:
            ax.set_title(f"H{hi}", fontsize=8)
plt.suptitle("All 144 GPT-2 attention heads on the test sentence", y=0.92)
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 4. Programmatically find a "previous-token" head

A previous-token head puts most of its attention on the **immediately preceding** position — high mass on the sub-diagonal of the attention matrix."""
    ),
    code(
        '''def previous_token_score(A):
    """Fraction of attention mass on the sub-diagonal (excluding row 0)."""
    T = A.shape[-1]
    sub = torch.tensor([A[i, i - 1].item() for i in range(1, T)])
    return sub.mean().item()

scores = []
for li in range(L):
    for hi in range(H):
        scores.append((previous_token_score(attns[li, hi]), li, hi))
scores.sort(reverse=True)

print("Top 5 previous-token heads:")
for s, l, h in scores[:5]:
    print(f"  L{l} H{h}: {s:.3f}")'''
    ),
    code(
        '''best_s, best_l, best_h = scores[0]
plt.figure(figsize=(6, 5))
plt.imshow(attns[best_l, best_h].numpy(), cmap='Blues')
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.yticks(range(len(tokens)), tokens)
plt.title(f"GPT-2 previous-token head: L{best_l} H{best_h} (sub-diag score {best_s:.3f})")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 5. Programmatically find a fixed-offset positional head"""
    ),
    code(
        '''def offset_k_score(A, k):
    T = A.shape[-1]
    return torch.tensor([A[i, i - k].item() for i in range(k, T)]).mean().item()

for k in [1, 2, 3]:
    best = max(((offset_k_score(attns[l, h], k), l, h) for l in range(L) for h in range(H)))
    print(f"best head for offset -{k}: L{best[1]} H{best[2]} score {best[0]:.3f}")'''
    ),
    md(
        """## 6. Find an **induction head**

Induction heads implement the pattern: if the prefix is `... A B ... A`, the head at the final `A` attends to the token *after* the earlier `A` (which is `B`), so the model can copy `B`. We'll feed a synthetic repeated-pattern sequence and look for heads with that behaviour.

(See *Olsson et al. 2022, "In-context Learning and Induction Heads"* for the full story.)"""
    ),
    code(
        '''# Build a sequence with a deliberately repeated pattern: tokens [a, b, c, d, a, ?]
# An induction head should attend, at the final 'a', to position 1 (i.e. 'b').
import random
random.seed(0)

# pick 4 random unique token ids in range [1000, 5000]
unique_ids = random.sample(range(1000, 5000), 4)
prefix = unique_ids + unique_ids        # length 8: [a, b, c, d, a, b, c, d]
input_ids = torch.tensor([prefix])
n = len(prefix)

with torch.no_grad():
    out = gpt2(input_ids, output_attentions=True)
A_all = torch.stack([a[0] for a in out.attentions])  # (L, H, n, n)

# For each head, at position 4 (second `a`), how much weight is on position 1 (the `b` after first `a`)?
induction = []
for li in range(L):
    for hi in range(H):
        score = A_all[li, hi, 4, 1].item()
        induction.append((score, li, hi))
induction.sort(reverse=True)

print("Top 5 induction-pattern heads (weight at pos 4 -> pos 1):")
for s, l, h in induction[:5]:
    print(f"  L{l} H{h}: {s:.3f}")'''
    ),
    code(
        '''s, li, hi = induction[0]
plt.figure(figsize=(5, 4))
plt.imshow(A_all[li, hi].numpy(), cmap='Blues')
plt.xticks(range(n), [f"p{i}" for i in range(n)])
plt.yticks(range(n), [f"p{i}" for i in range(n)])
plt.title(f"Induction-like head L{li} H{hi}\\nweight(p4 -> p1) = {s:.3f}")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 7. Compare against multi-head versions of our own implementation

A sanity check: our `MultiHeadAttention` from \u00a71 should produce the same output as running `n_heads` independent single-head attentions (modulo the shared `W_qkv` packing)."""
    ),
    code(
        '''class SingleHeadForCheck(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_head, bias=False)
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_head, bias=False)
    def forward(self, x, causal=True):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        T = x.size(-2)
        sc = (Q @ K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            sc = sc.masked_fill(~mask, float("-inf"))
        w = F.softmax(sc, dim=-1)
        return w @ V, w

D, H_, Dh = 32, 4, 8
x = torch.randn(1, 5, D)
mha = MultiHeadAttention(d_model=D, n_heads=H_)

# Force MHA's W_qkv to use known per-head slices so we can replicate them
heads = [SingleHeadForCheck(D, Dh) for _ in range(H_)]
with torch.no_grad():
    # MHA's W_qkv weight is (3D, D). Layout: rows 0..D Q, D..2D K, 2D..3D V.
    Wq, Wk, Wv = mha.W_qkv.weight.chunk(3, dim=0)
    for h in range(H_):
        heads[h].Wq.weight.copy_(Wq[h * Dh:(h + 1) * Dh])
        heads[h].Wk.weight.copy_(Wk[h * Dh:(h + 1) * Dh])
        heads[h].Wv.weight.copy_(Wv[h * Dh:(h + 1) * Dh])

with torch.no_grad():
    mha_out, _ = mha(x)
    single_outs = [h(x)[0] for h in heads]
    cat = torch.cat(single_outs, dim=-1)
    cat = mha.W_o(cat)

print(f"max abs diff between MHA and stacked single heads: {(mha_out - cat).abs().max().item():.2e}")'''
    ),
    md(
        """## 8. What you just did

- Implemented multi-head attention as one packed tensor op (the form used by every real LLM).
- Verified equivalence with `n_heads` separate single-head attentions.
- Dumped every one of GPT-2's 144 attention patterns and visually identified specializations.
- **Algorithmically** found a previous-token head, a fixed-offset positional head, and an induction-like head.

This is the picture you want in your head for the rest of the series: a transformer is a stack of layers, each of which runs *several* attention heads in parallel, each looking for a different relationship in the residual stream.

**Next: Post 6 \u2014 Positional information: from sinusoids to RoPE.**"""
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
