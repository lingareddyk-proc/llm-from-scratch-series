# Post 5 — Multi-Head Attention & Head Specialization

**Series:** LLM From Scratch · Post **5 of 12** · *Act II — Build a modern decoder*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/05-multi-head-attention/notebook.ipynb)

## 60-second summary

A single attention head can compute one weighted average; a real transformer runs many in parallel so each head can specialize. This post implements multi-head attention as one packed tensor op (the form every real LLM uses) and then dumps all **144 GPT-2 attention heads** on a real sentence. It then *algorithmically* identifies a **previous-token head**, a **fixed-offset positional head**, and an **induction head** — the building blocks of in-context learning. Closes with the GQA / MQA / MHA comparison table for current frontier models.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                                     |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~10 min on free Colab CPU.                               |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- Why multiple heads beat one big head
- Packed `(B, H, T, d_head)` implementation
- Equivalence with stacked single-head attentions
- Head zoo: previous-token, positional, induction, coreference
- Algorithmically discovering each type in pretrained GPT-2
- MHA vs GQA vs MQA — table of what frontier models use today
