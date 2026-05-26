"""Generate notebook.ipynb for Post 4 — Attention From First Principles."""

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
        """# Post 4 — Attention From First Principles

Companion notebook to the Medium post. Runs in ~15 min on free Colab CPU.

We will:

1. Implement **scaled dot-product attention** in NumPy.
2. Re-implement it in PyTorch with learnable Q/K/V projections.
3. Visualize attention heatmaps on a real sentence — first untrained, then with a real attention head from pretrained GPT-2.
4. Apply the causal mask and watch the upper triangle disappear.
5. Show that for the word `it` in *"The cat sat on the mat because it was tired"*, GPT-2 attention heads really do attend to `cat`."""
    ),
    md("## 0. Setup"),
    code('!pip -q install "transformers>=4.44" "torch>=2.3" matplotlib'),
    code(
        """import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
np.random.seed(0)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")  # eager so we can read attn weights
gpt2.eval()
print("Loaded GPT-2 with eager attention (so we can inspect attention weights).")"""
    ),
    md(
        """## 1. Scaled dot-product attention in NumPy (the whole thing in ~10 lines)"""
    ),
    code(
        '''def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention_np(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)                      # (seq, seq)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax_np(scores, axis=-1)              # (seq, seq), each row sums to 1
    out = weights @ V                                  # (seq, d)
    return out, weights


seq, d = 5, 8
Q = np.random.randn(seq, d)
K = np.random.randn(seq, d)
V = np.random.randn(seq, d)
out, w = attention_np(Q, K, V)
print("output shape:", out.shape)
print("each row of weights sums to 1?", np.allclose(w.sum(axis=-1), 1.0))
print("weights:\\n", np.round(w, 3))'''
    ),
    md(
        """## 2. Add the causal mask

`mask[i, j] = True` iff token `i` is allowed to attend to token `j`. The lower-triangular indicator does exactly that."""
    ),
    code(
        """causal = np.tril(np.ones((seq, seq)) == 1)
print("causal mask (True = allowed):")
print(causal.astype(int))

out_c, w_c = attention_np(Q, K, V, mask=causal)
print("\\ncausal weights:\\n", np.round(w_c, 3))
print("\\nNote: row 0 attends only to position 0; row 4 attends to all positions 0..4.")"""
    ),
    md("## 3. Visualize the mask vs the attention matrix"),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(w, vmin=0, vmax=1, cmap='Blues')
axes[0].set_title("attention weights (no mask)")
axes[1].imshow(w_c, vmin=0, vmax=1, cmap='Blues')
axes[1].set_title("attention weights (causal mask)")
for ax in axes:
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
plt.tight_layout(); plt.show()"""
    ),
    md(
        """## 4. PyTorch implementation with learnable projections

Now we wrap it as an `nn.Module`. This is what one attention head inside a real transformer looks like (modulo multi-head and RoPE — coming in Posts 5 and 6)."""
    ),
    code(
        '''class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, causal: bool = True):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.W_o = nn.Linear(d_head, d_model, bias=False)
        self.d_head = d_head
        self.causal = causal

    def forward(self, x):
        Q = self.W_q(x)  # (B, T, d_head)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5)  # (B, T, T)

        if self.causal:
            T = x.size(-2)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V                   # (B, T, d_head)
        return self.W_o(out), weights       # (B, T, d_model), (B, T, T)


x = torch.randn(1, 6, 32)  # (batch=1, seq=6, d_model=32)
attn = SingleHeadAttention(d_model=32, d_head=16, causal=True)
out, w = attn(x)
print("out shape:    ", tuple(out.shape))
print("weights shape:", tuple(w.shape))
print("row sums (should all be 1):", w[0].sum(dim=-1).tolist())'''
    ),
    md(
        """## 5. Attention on a real sentence — random weights first

Let's encode our cat-mat-tired sentence with GPT-2's embeddings and feed it through our **untrained** attention head. The pattern won't be meaningful (random weights), but it confirms the plumbing."""
    ),
    code(
        '''sentence = "The cat sat on the mat because it was tired"
ids = tokenizer.encode(sentence)
tokens = [tokenizer.decode([i]).strip() or "_" for i in ids]
print("tokens:", tokens)

x = gpt2.transformer.wte(torch.tensor([ids]))  # (1, T, 768)
attn = SingleHeadAttention(d_model=768, d_head=64, causal=True)
with torch.no_grad():
    _, weights = attn(x)
plt.figure(figsize=(6, 5))
plt.imshow(weights[0].numpy(), cmap='Blues', vmin=0, vmax=weights[0].max().item())
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.yticks(range(len(tokens)), tokens)
plt.title("Untrained attention head (random projections)")
plt.xlabel("attending to (key)")
plt.ylabel("from (query)")
plt.tight_layout(); plt.show()
print("Note the lower-triangular structure from the causal mask.")'''
    ),
    md(
        """## 6. A *trained* head: what does GPT-2 actually attend to for the word `it`?

Now we pull real attention weights from pretrained GPT-2. We'll look at every layer × every head for the query position corresponding to ` it`, and find the head whose attention is most concentrated on ` cat`. (Spoiler: there will be one.)"""
    ),
    code(
        '''with torch.no_grad():
    out = gpt2(torch.tensor([ids]), output_attentions=True)

attns = out.attentions  # tuple of (n_layer,) each (B, n_heads, T, T)
print(f"layers: {len(attns)}, heads/layer: {attns[0].shape[1]}, T={attns[0].shape[-1]}")

it_idx  = tokens.index("it")
cat_idx = tokens.index("cat")

best = (-1.0, None, None)
for L, A in enumerate(attns):
    A = A[0]  # (n_heads, T, T)
    for h in range(A.shape[0]):
        w_it = A[h, it_idx]            # (T,)  what does `it` attend to in this head?
        score = w_it[cat_idx].item()    # how much of that attention goes to `cat`?
        if score > best[0]:
            best = (score, L, h)

print(f"\\nMost cat-focused head for `it`: layer={best[1]}, head={best[2]}, weight on `cat` = {best[0]:.3f}")'''
    ),
    code(
        '''L, h = best[1], best[2]
A = attns[L][0, h].numpy()

plt.figure(figsize=(7, 5))
plt.imshow(A, cmap='Blues', vmin=0, vmax=A.max())
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.yticks(range(len(tokens)), tokens)
plt.title(f"GPT-2 layer {L}, head {h} \u2014 trained attention head")
plt.xlabel("attending to (key)")
plt.ylabel("from (query)")

plt.gca().add_patch(plt.Rectangle((cat_idx - 0.5, it_idx - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
plt.tight_layout(); plt.show()

print("Red box highlights the (query=`it`, key=`cat`) cell.")
print("Notice: from row `it`, a lot of probability mass lands on `cat`.")'''
    ),
    md(
        """## 7. What `it` attends to, plotted as a 1-D bar chart"""
    ),
    code(
        '''w_it = attns[L][0, h, it_idx].numpy()
plt.figure(figsize=(8, 3))
plt.bar(range(len(tokens)), w_it, color=['red' if t == 'cat' else 'steelblue' for t in tokens])
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.ylabel("attention weight")
plt.title(f"What `it` attends to in GPT-2 layer {L}, head {h}")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 8. Effect of `d_head` and "temperature" on attention sharpness

Smaller `d_head` (smaller denominator under √d) or higher pre-softmax scale → sharper attention. Larger → softer."""
    ),
    code(
        '''def attention_sharpness_demo(d_head, scale=1.0):
    Q = torch.randn(8, d_head)
    K = torch.randn(8, d_head)
    scores = Q @ K.T / (d_head ** 0.5) * scale
    w = F.softmax(scores, dim=-1)
    return w.numpy()

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, scale in zip(axes, [0.5, 1.0, 3.0]):
    w = attention_sharpness_demo(d_head=16, scale=scale)
    ax.imshow(w, vmin=0, vmax=1, cmap='Blues')
    ax.set_title(f"scale x{scale}")
plt.suptitle("Same Q,K — sharper attention as you scale scores up")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 9. What you just did

- Wrote scaled dot-product attention in 10 lines of NumPy.
- Wrapped it as a PyTorch module with learnable Q/K/V/O projections.
- Verified the causal mask zeroes the upper triangle.
- Found a real GPT-2 attention head that — given the prompt *"The cat sat on the mat because it was tired"* — attends from `it` to `cat`.

That last point is the punchline of mechanistic interpretability: attention heads **specialize**, and modern transformers (Gemini, Claude) build everything on top of this mechanism.

**Next: Post 5 \u2014 Multi-head attention and why heads specialize.** We'll explore *all* the heads, not just the one that did the cat-it lookup."""
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
