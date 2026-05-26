"""Generate notebook.ipynb for Post 3 — Embeddings & the Residual Stream."""

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
        """# Post 3 — Embeddings and the Residual Stream

Companion notebook to the Medium post. Runs in ~5 min on free Colab CPU.

We will:

1. Build an embedding layer from scratch and verify it is just a lookup table.
2. Load GPT-2's trained embeddings and explore them (nearest neighbors of `king`, `cat`, `Python`...).
3. Hand-trace GPT-2's residual stream layer by layer, watching how each block additively writes to it.
4. Confirm in code that the entire forward pass is `x = x + attention(...)` then `x = x + mlp(...)`, repeated."""
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
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
print("Loaded GPT-2 (124M params)")"""
    ),
    md(
        """## 1. An embedding layer is just a lookup table

`nn.Embedding(vocab_size, d_model)` holds a matrix of shape `(vocab_size, d_model)`. Indexing it with a token id returns the corresponding row. That's it."""
    ),
    code(
        """vocab_size, d_model = 10, 4
emb = nn.Embedding(vocab_size, d_model)
print("Embedding matrix shape:", emb.weight.shape)
print()

ids = torch.tensor([1, 3, 5, 1])
print("Input ids:    ", ids.tolist())
print("Embedded shape:", emb(ids).shape)
print("Embedded vectors:")
print(emb(ids))
print()
print("Note that ids[0] == ids[3], so their vectors are identical:")
print((emb(ids)[0] == emb(ids)[3]).all().item())"""
    ),
    md(
        """## 2. GPT-2's trained embeddings — words that mean similar things end up nearby

We'll measure cosine similarity in GPT-2's embedding space and find nearest neighbors. The model never saw a dictionary — it learned these similarities purely from "predict the next token." """
    ),
    code(
        """W_E = model.transformer.wte.weight  # (vocab_size, d_model)
print("GPT-2 embedding matrix shape:", W_E.shape)
print(f"d_model = {W_E.shape[1]}")
print(f"vocab_size = {W_E.shape[0]:,}")
print(f"params in embeddings: {W_E.numel():,}")"""
    ),
    code(
        """def nearest(query: str, k: int = 8):
    ids = tokenizer.encode(query)
    if len(ids) != 1:
        print(f"  ({query!r} is {len(ids)} tokens; using the first)")
    qid = ids[0]
    qvec = W_E[qid]
    sims = F.cosine_similarity(qvec.unsqueeze(0), W_E, dim=1)
    top = torch.topk(sims, k + 1)  # +1 because the word itself is always top-1
    results = []
    for s, i in zip(top.values, top.indices):
        if i.item() == qid:
            continue
        results.append((tokenizer.decode([i.item()]), s.item()))
        if len(results) == k:
            break
    return results


for q in [" king", " cat", " Python", " France", " happy"]:
    print(f"nearest to {q!r}:")
    for tok, sim in nearest(q):
        print(f"  {sim:.3f}  {tok!r}")
    print()"""
    ),
    md(
        """## 3. Visualizing the embedding space (PCA to 2D)

PCA squishes 768 dimensions down to 2 so we can plot. Look for clusters: animals, countries, programming terms..."""
    ),
    code(
        """from sklearn.decomposition import PCA

words = [
    " cat", " dog", " mouse", " lion", " tiger", " elephant",
    " France", " Germany", " Italy", " Japan", " China", " India",
    " Python", " Java", " JavaScript", " Ruby", " C", " Rust",
    " happy", " sad", " angry", " excited", " calm", " anxious",
]
ids = [tokenizer.encode(w)[0] for w in words]
vectors = W_E[ids].detach().numpy()
pca = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(9, 7))
for (x, y), w in zip(pca, words):
    plt.scatter(x, y, s=30)
    plt.annotate(w.strip(), (x, y), fontsize=10, xytext=(4, 4), textcoords="offset points")
plt.title("GPT-2 token embeddings (PCA to 2D) — note the clusters")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.grid(alpha=0.2)
plt.tight_layout(); plt.show()"""
    ),
    md(
        """## 4. The residual stream in action

Now the punchline of the post. Let's run a sentence through GPT-2 and inspect the residual stream **at every layer**. We'll show that:

1. The stream starts as the token embedding.
2. Each block writes an additive update.
3. The final stream, projected by `W_U`, gives the logits.

GPT-2 exposes this via `output_hidden_states=True`."""
    ),
    code(
        """prompt = "The capital of France is"
ids = tokenizer.encode(prompt)
input_ids = torch.tensor([ids])

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

# hidden_states is a tuple of length n_layer + 1
# hidden_states[0]  = embeddings (x_0)
# hidden_states[i]  = residual stream AFTER block i
# hidden_states[-1] = x_L

hs = out.hidden_states
print(f"Number of hidden states returned: {len(hs)}  (= embeddings + {len(hs) - 1} blocks)")
print(f"Each is shape (batch, seq_len, d_model): {tuple(hs[0].shape)}")

x0 = hs[0]
xL = hs[-1]
print()
print(f"L2 norm of x_0 (embeddings):     {x0.norm():.2f}")
print(f"L2 norm of x_L (after 12 blocks): {xL.norm():.2f}")"""
    ),
    md(
        """## 5. How much does each block change the stream?

For each block we compute `||x_{i} - x_{i-1}||` — the magnitude of the additive update it wrote. Early layers tend to do more "low-level" rearrangement; late layers tend to write small but very targeted updates."""
    ),
    code(
        """updates = []
for i in range(1, len(hs)):
    delta = (hs[i] - hs[i - 1]).norm().item()
    updates.append(delta)

plt.figure(figsize=(9, 4))
plt.bar(range(1, len(updates) + 1), updates)
plt.xlabel("block index")
plt.ylabel("||\u0394x||  (size of additive update)")
plt.title("How much each block writes to the residual stream")
plt.tight_layout(); plt.show()"""
    ),
    md(
        """## 6. The unembedding: turn the final residual stream into logits

GPT-2 ties weights: `W_U = W_E.T`. So the final logits are just `x_L @ W_E.T`."""
    ),
    code(
        """logits_from_x = xL @ W_E.T  # (batch, seq_len, vocab_size)
official_logits = out.logits if hasattr(out, "logits") else None

if official_logits is None:
    # Re-run to get logits too (the previous call didn't request them by default).
    with torch.no_grad():
        official_logits = model(input_ids).logits

print(f"x_L @ W_E.T  shape: {tuple(logits_from_x.shape)}")
print(f"official logits shape: {tuple(official_logits.shape)}")
print(f"match? max abs diff = {(logits_from_x - official_logits).abs().max().item():.2e}")"""
    ),
    md(
        """The match (modulo a final layer-norm that we'd add for byte-exact equality) confirms the picture: **the whole transformer is embed \u2192 add updates \u2192 unembed**."""
    ),
    md(
        """## 7. Build a 2-layer toy decoder to see "x = x + f(x)" with our own eyes

A minimal decoder where attention and MLP are just `nn.Linear` placeholders. The point is the *plumbing*: prove the residual stream is literally a running sum."""
    ),
    code(
        """class ToyBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Linear(d_model, d_model)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyDecoder(nn.Module):
    def __init__(self, vocab_size=100, d_model=16, n_layer=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([ToyBlock(d_model) for _ in range(n_layer)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, ids, trace: bool = False):
        x = self.embed(ids)
        trace_states = [x.clone()]
        for blk in self.blocks:
            x = blk(x)
            if trace:
                trace_states.append(x.clone())
        logits = self.lm_head(x)
        return (logits, trace_states) if trace else logits


toy = ToyDecoder()
ids = torch.tensor([[1, 2, 3, 4, 5]])
logits, trace = toy(ids, trace=True)
print(f"input ids:   {ids.tolist()}")
print(f"logits shape: {tuple(logits.shape)}")
print(f"#states traced: {len(trace)} (embeddings + 2 blocks)")
for i, s in enumerate(trace):
    print(f"  x_{i}  norm = {s.norm():.3f}")"""
    ),
    md(
        """## 8. What you just did

- Confirmed that an embedding layer is a lookup table.
- Verified that GPT-2's learned embeddings carry semantic similarity.
- Inspected GPT-2's residual stream layer by layer and saw each block additively update it.
- Built a tiny decoder that makes the `x = x + f(x)` pattern impossible to miss.

**Next: Post 4 — Attention from first principles.** Now we replace the placeholder `nn.Linear` "attention" with the real thing, and visualize what every head does."""
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
