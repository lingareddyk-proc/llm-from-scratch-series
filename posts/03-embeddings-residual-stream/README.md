# Post 3 — Embeddings and the Residual Stream

**Series:** LLM From Scratch · Post **3 of 12** · *Act I — Foundations*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/03-embeddings-residual-stream/notebook.ipynb)

## 60-second summary

Tokens become vectors via a lookup table — the embedding layer. Those vectors live in a shared scratchpad called the **residual stream**, and every block in a transformer just adds an update to it. This post proves the picture in code: it explores GPT-2's learned embeddings (PCA cluster plot + nearest-neighbour search), traces the residual stream layer by layer through a real forward pass, and confirms that `x_L @ W_E.T` reproduces the model's logits — i.e. the whole transformer really is *embed → add updates → unembed*.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                                     |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~5 min on free Colab CPU.                                |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook. Re-run to regenerate the `.ipynb`.      |

## Concepts covered

- Embedding layers as lookup tables
- Cosine-similarity geometry of learned embeddings (`king`, `cat`, `Python`, ...)
- The **residual stream** — every block adds, never replaces
- Per-block update magnitude (`||Δx||`) across all 12 GPT-2 blocks
- Tied vs untied output weights (`W_U = W_E.T`)
- Why the residual stream is the right mental model for every later post
