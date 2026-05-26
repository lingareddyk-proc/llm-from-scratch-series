# Post 6 — Positional Encoding & RoPE

**Series:** LLM From Scratch · Post **6 of 12** · *Act II — Build a modern decoder*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/06-positional-rope/notebook.ipynb)

## 60-second summary

Attention is permutation-invariant — without positional information the model can't tell *"cat saw dog"* from *"dog saw cat"*. This post walks through the three answers the field has used: **absolute sinusoidal** (2017), **learned absolute** (GPT-2), and **RoPE** — the rotary position embedding that Llama 3, Mistral, Gemma, and (per public statements) Gemini and Claude all use today. Implements RoPE in ~15 lines, visualizes its 2-D rotation, and proves numerically that `<RoPE(q,m), RoPE(k,n)>` depends only on `n-m`. Ends by dropping RoPE into the multi-head attention from Post 5 as a drop-in replacement.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                                     |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~5 min on free Colab CPU.                                |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- Why attention is permutation-invariant without positions
- Sinusoidal vs learned vs rotary positional encoding
- RoPE rotation in 2-D subspaces (visualized)
- Relative-position property: dot products constant along diagonals
- RoPE as a drop-in for multi-head attention
- Long-context extension intuition (NTK / YaRN)
