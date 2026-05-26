# Post 10 — Mixture of Experts

**Series:** LLM From Scratch · Post **10 of 12** · *Act III — Frontier techniques*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/10-mixture-of-experts/notebook.ipynb)

## 60-second summary

GPT-4, Gemini 1.5/2.x/3, Mixtral, DeepSeek V3, Grok — the modern frontier is dominated by **Mixture of Experts**. This post replaces our `TinyLLM`'s SwiGLU FFN with a small **4-expert top-1 MoE** block (gate + experts + Switch-style load-balance loss), trains it briefly on TinyStories, and **visualizes which expert each token gets routed to** in every layer. Closes with an active-vs-total parameter accounting that explains why MoE gives more capability per FLOP.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article with diagram and frontier MoE table.                 |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~15 min on free Colab.                                   |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- Token-level routing (gate + top-k)
- Load-balancing auxiliary loss (Switch Transformer style)
- Token-choice vs expert-choice routing
- Capacity, dropping, expert parallelism (named, not implemented)
- Active params vs total params (and why they're advertised separately)
- Frontier MoE topologies: Mixtral 8×7B, DeepSeek V3 256-expert, GPT-4 ~16-expert
