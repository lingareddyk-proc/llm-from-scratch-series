# Post 4 — Attention From First Principles

**Series:** LLM From Scratch · Post **4 of 12** · *Act I — Foundations*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/04-attention-from-first-principles/notebook.ipynb)

## 60-second summary

The keystone post of the series. Builds **scaled dot-product attention** — `softmax(Q Kᵀ / √d) V` — from a blank slate, first in NumPy, then in PyTorch with learnable Q/K/V projections. Visualizes the causal mask, then loads pretrained GPT-2 and **finds the attention head that resolves the word ` it` to ` cat`** in the classic sentence *"The cat sat on the mat because it was tired."* This is the same equation that runs inside Claude and Gemini — every later post is engineering on top of it.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                                     |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~15 min on free Colab CPU.                               |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- Attention as soft, content-dependent weighted average
- Q / K / V — the dictionary analogy
- Scaled dot product and the role of √d
- Causal masking (the only difference between decoder and encoder)
- Hands-on: find the real GPT-2 head that resolves *it → cat*
- Effect of pre-softmax scale on attention sharpness
- Mental model for every later post in the series
