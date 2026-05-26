# Post 11 — From Base Model to Assistant: SFT and DPO

**Series:** LLM From Scratch · Post **11 of 12** · *Act III — Frontier techniques*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/11-sft-and-dpo/notebook.ipynb)

## 60-second summary

A base model can complete text, not follow instructions. This post does the post-training that turns a base model into an assistant — the same two-stage recipe behind ChatGPT, Claude, and Gemini: **SFT** (supervised fine-tuning with masked cross-entropy on assistant tokens) and **DPO** (Direct Preference Optimization — the modern simpler alternative to RLHF/PPO). Builds a toy "say hello in <language>" instruction dataset that our 6M-parameter model can actually learn, then SFTs, then DPOs against hand-crafted preference pairs, and verifies the chosen vs rejected log-likelihood margin moves in the right direction.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article with three-stage diagram and frontier-recipe notes.  |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — ~10 min on free Colab.                                   |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Concepts covered

- Base vs SFT vs aligned models
- Masked cross-entropy on assistant tokens
- Chat templates and special tokens
- RLHF / PPO (sketched)
- DPO loss in closed form
- The (generate → annotate → DPO → repeat) frontier loop

## Reuses & is reused by

- **Reuses:** `src/model.py` (Post 7) and the Post 8 checkpoint.
- **Reused by:** Post 12's frontier-mapping summary.
