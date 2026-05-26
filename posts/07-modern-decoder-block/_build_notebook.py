"""Generate notebook.ipynb for Post 7 — Full modern decoder block."""

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
        """# Post 7 — The Full Modern Decoder Block (Llama / Gemma Style)

Companion notebook to the Medium post. ~5 min on free Colab CPU.

We will:

1. Pull in `src/model.py` from the series repo \u2014 the canonical small modern LLM we'll reuse for the rest of the series.
2. Inspect every class: `RMSNorm`, `GroupedQueryAttention` with RoPE, `SwiGLU`, `DecoderBlock`, `TinyLLM`.
3. Instantiate the default config (~10M parameters) and trace shapes through a forward pass.
4. Visualize **gradient flow** across blocks (Pre-Norm should give nice flat gradients).
5. Compare parameter counts between a 2017-style block and our modern block at the same `d_model`."""
    ),
    md("## 0. Setup \u2014 clone the series repo and import"),
    code(
        '''!pip -q install "torch>=2.3" matplotlib
import os, sys, subprocess

REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("ok")'''
    ),
    code(
        """import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import (
    ModelConfig,
    RMSNorm,
    GroupedQueryAttention,
    SwiGLU,
    DecoderBlock,
    TinyLLM,
    build_rope_cache,
    apply_rope,
)

torch.manual_seed(0)"""
    ),
    md(
        """## 1. RMSNorm \u2014 LayerNorm minus the mean"""
    ),
    code(
        '''rmsn = RMSNorm(d=32)
x = torch.randn(4, 32)
y = rmsn(x)
print(f"input rms:  {x.pow(2).mean(-1).sqrt().tolist()}")
print(f"output rms (should be ~1): {y.pow(2).mean(-1).sqrt().tolist()}")'''
    ),
    md(
        """## 2. GroupedQueryAttention with RoPE \u2014 fewer K/V heads than Q heads

We saw the math in Posts 4-6. Now it's all packaged into one class. Default config: 8 Q-heads, 2 KV-heads (GQA ratio 4)."""
    ),
    code(
        '''cfg = ModelConfig()
print(cfg)
print(f"d_head = {cfg.d_head}, GQA ratio = {cfg.n_kv_groups}")
gqa = GroupedQueryAttention(cfg)
x = torch.randn(2, 16, cfg.d_model)
y = gqa(x)
print(f"input  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}  (matches input \u2014 ready to add to residual)")'''
    ),
    md(
        """## 3. SwiGLU \u2014 gate \u00d7 up, then project down"""
    ),
    code(
        '''ffn = SwiGLU(d_model=cfg.d_model, d_ff=cfg.d_ff)
print(f"params in SwiGLU: {sum(p.numel() for p in ffn.parameters()):,}")
print(f"vs equivalent ReLU-FFN (2 matrices of width 4*d_model): {2 * cfg.d_model * (4 * cfg.d_model):,}")
y = ffn(x)
print(f"output shape: {tuple(y.shape)}")'''
    ),
    md(
        """## 4. The full decoder block: Pre-Norm + GQA + Pre-Norm + SwiGLU

Same as the diagram in the article. Two `x = x + f(norm(x))` lines."""
    ),
    code(
        '''block = DecoderBlock(cfg)
y = block(x)
print(f"output shape: {tuple(y.shape)}")

print("\\nBlock contents:")
for name, mod in block.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name:<6} {mod.__class__.__name__:<22} params: {n:>9,}")'''
    ),
    md(
        """## 5. The full model: embedding + N blocks + RMSNorm + tied lm_head"""
    ),
    code(
        '''model = TinyLLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}  ({n_params / 1e6:.2f} M)")

ids = torch.randint(0, cfg.vocab_size, (2, 64))
logits = model(ids)
print(f"\\ninput ids shape:    {tuple(ids.shape)}")
print(f"output logits shape: {tuple(logits.shape)}  (B, T, vocab_size)")'''
    ),
    md(
        """## 6. Trace shapes through the forward pass

Confirm each stage outputs the shape we expect."""
    ),
    code(
        '''x = model.tok_emb(ids); print(f"after embedding:  {tuple(x.shape)}")
for i, blk in enumerate(model.blocks):
    x = blk(x); print(f"after block {i}:     {tuple(x.shape)}")
x = model.norm_f(x); print(f"after final norm: {tuple(x.shape)}")
logits = x @ model.lm_head_weight.T
print(f"after lm_head:    {tuple(logits.shape)}")'''
    ),
    md(
        """## 7. Visualize gradient flow

Pre-Norm should give gradients that don't blow up or vanish across depth. We compute `||\u2202loss/\u2202x_layer||` for each block."""
    ),
    code(
        '''import torch.autograd as autograd

ids = torch.randint(0, cfg.vocab_size, (1, 64))
targets = torch.randint(0, cfg.vocab_size, (1, 64))

x = model.tok_emb(ids).detach().clone().requires_grad_(True)
xs = [x]
for blk in model.blocks:
    xs.append(blk(xs[-1]))
final = model.norm_f(xs[-1])
logits = final @ model.lm_head_weight.T
loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

# Compute gradient norms w.r.t. every block output
grads = autograd.grad(loss, xs, retain_graph=False)
norms = [g.norm().item() for g in grads]

plt.figure(figsize=(8, 4))
plt.bar(range(len(norms)), norms)
plt.xlabel("layer index (0 = embeddings)")
plt.ylabel("||\u2202loss/\u2202x_layer||")
plt.title("Gradient flow through the Pre-Norm decoder \u2014 smooth across depth")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 8. Side-by-side parameter count: 2017 block vs modern block

Same `d_model`, same number of params \u2014 the modern block spends its parameters differently (SwiGLU has 3 matrices, GQA shrinks KV)."""
    ),
    code(
        '''def post_norm_block_params(d_model, n_heads, d_ff):
    # 2017-style: Multi-Head Attention (Wq,Wk,Wv,Wo all d_model->d_model with bias)
    # FFN: Linear(d_model, d_ff) + Linear(d_ff, d_model), with bias, ReLU.
    qkvo = 4 * (d_model * d_model + d_model)
    ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    ln1 = 2 * d_model  # gamma + beta
    ln2 = 2 * d_model
    return qkvo + ffn + ln1 + ln2

orig_params = post_norm_block_params(cfg.d_model, cfg.n_heads_q, cfg.d_ff)
modern_params = sum(p.numel() for p in DecoderBlock(cfg).parameters())
print(f"2017-style block  (LN + MHA + bias + ReLU-FFN): {orig_params:>9,}")
print(f"Modern block      (RMSNorm + GQA + SwiGLU):    {modern_params:>9,}")
print(f"\\nFor the same total budget the modern block has a slightly larger FFN (SwiGLU has 3 matrices)")
print("and a slightly smaller attention block (no biases, GQA shrinks K/V).")'''
    ),
    md(
        """## 9. What you just did

- Imported the canonical `src/model.py` that every later post will reuse.
- Inspected every component: RMSNorm, GQA+RoPE, SwiGLU, DecoderBlock, TinyLLM.
- Forwarded a real input batch shape-by-shape.
- Verified Pre-Norm gives smooth gradient flow across depth.
- Compared parameter budgets with the 2017-style block.

**Next: Post 8 \u2014 Pretrain this model from scratch on TinyStories in ~30 minutes on free Colab T4.**"""
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
