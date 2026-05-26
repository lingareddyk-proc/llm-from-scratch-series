# Post 12 — Putting It Together: How a Frontier LLM Is Actually Built

**Series:** LLM From Scratch · Post **12 of 12** · *Act III — Frontier techniques*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/12-frontier-llm-mapping/notebook.ipynb)

## 60-second summary

The capstone of the series. Maps every component built across Posts 1–11 onto the public blueprint of a frontier LLM (Gemini 3, Claude 4, GPT-4o, Llama 3/4) with a master comparison table. Names what was deliberately *not* covered — multimodality, long-context tricks, FlashAttention, PagedAttention, speculative decoding, distillation, Constitutional AI — and where to go to learn each. Ships one combined notebook that imports `src/model.py`, builds the full modern `TinyLLM`, prints a class-by-class component inventory, optionally pretrains, generates with the KV-cache, and prints the side-by-side architecture comparison vs Llama 3 8B.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article with the master comparison table and reading list.   |
| [`notebook.ipynb`](notebook.ipynb)         | Capstone Colab notebook — ~15 min, free Colab.                            |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- One-paragraph summary of the entire architecture
- Master comparison: ours vs Llama 3 8B/70B vs Mixtral 8×7B vs frontier
- What we deliberately did *not* cover (with where to learn each)
- Reading list (foundational, architecture, training/alignment, inference)
- Three concrete next steps (scale up, read nanoGPT, pick a frontier topic)

## Reuses

- `src/model.py` (Post 7), `src/train.py` (Post 8), Post 8 checkpoint, Post 9 KV-cache pattern, Post 11 fine-tuning recipe.
