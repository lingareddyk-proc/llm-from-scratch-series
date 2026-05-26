# Post 7 — The Full Modern Decoder Block (Llama / Gemma Style)

**Series:** LLM From Scratch · Post **7 of 12** · *Act II — Build a modern decoder*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/07-modern-decoder-block/notebook.ipynb)

## 60-second summary

Wires every piece from Posts 3–6 into the actual decoder block used by Llama 3 and Gemma 2: **Pre-Norm + RMSNorm + Grouped-Query Attention with RoPE + SwiGLU**. Ships [`src/model.py`](../../src/model.py) — the small (~10M-param) `TinyLLM` reused by Posts 8–11. The notebook inspects every class, traces shapes through a forward pass, visualizes gradient flow (Pre-Norm gives smooth gradients across depth), and contrasts parameter budgets with the original 2017 block.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article with side-by-side 2017 vs 2025 diagram.              |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~5 min on free Colab CPU.                                |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Repo additions

- New module: [`src/model.py`](../../src/model.py) — `RMSNorm`, `GroupedQueryAttention`, `SwiGLU`, `DecoderBlock`, `TinyLLM`, `ModelConfig`. **Imported by Posts 8–11.**

## Concepts covered

- Pre-Norm vs Post-Norm residuals
- RMSNorm vs LayerNorm
- Grouped-Query Attention (GQA)
- RoPE inside attention (no positional adds)
- SwiGLU FFN vs ReLU FFN
- Gradient flow visualization across depth
- 2017 vs 2025 block parameter accounting
