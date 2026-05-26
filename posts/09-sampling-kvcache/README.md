# Post 9 — Sampling and the KV-Cache

**Series:** LLM From Scratch · Post **9 of 12** · *Act III — Frontier techniques*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/09-sampling-kvcache/notebook.ipynb)

## 60-second summary

Naïve generation is O(T²) — every new token re-runs attention over the whole prefix. The **KV-cache** turns that into O(T) by caching the K and V vectors for past positions and only computing fresh Q for the new token. This post implements a `KVCachedTinyLLM` from scratch, loads the checkpoint trained in Post 8, and **benchmarks naïve vs cached generation** — typically a 5–30× speedup. Then covers the modern sampling toolkit (temperature, top-k, top-p, **min-p**) with side-by-side comparisons on the same prompt.

## What's in this folder

| File                                       | What it is                                                         |
| ------------------------------------------ | ------------------------------------------------------------------ |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                              |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~10 min, runs on CPU or T4.                       |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                  |

## Concepts covered

- Why naïve autoregressive generation is O(T²)
- The KV-cache: cache K and V, recompute only Q
- Why it's "KV" not "QKV"
- Cache-size implications and how GQA shrinks them
- Top-k vs top-p (nucleus) vs min-p
- Speculative decoding (preview)
- PagedAttention, continuous batching, KV quantization (named, not built)

## Reuses & is reused by

- **Reuses:** `src/model.py` (Post 7) and the Post 8 checkpoint.
- **Reused by:** Post 11 uses the cached generation function for its SFT chat demos.
