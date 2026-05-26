# Post 8 — Pretraining a Small LLM End-to-End on Colab

**Series:** LLM From Scratch · Post **8 of 12** · *Act II — Build a modern decoder*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/08-pretraining-tinystories/notebook.ipynb)

## 60-second summary

Pretrains the `TinyLLM` from Post 7 on **TinyStories**, end-to-end, on a free Colab T4 GPU in ~30 minutes. ~6.6M parameters, ~13M tokens, 2000 steps, AdamW + cosine LR + warmup + grad clip + bfloat16 — every dial frontier labs turn, just smaller. The notebook samples a fixed prompt at every eval checkpoint so the reader **watches the model go from gibberish at step 0 to coherent short stories at step 2000**.

## What's in this folder

| File                                       | What it is                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article with training-recipe table and mermaid loop.         |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — **needs a T4 GPU**, ~30 min run.                         |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook.                                         |

## Repo additions

- New module: [`src/train.py`](../../src/train.py) — `TrainConfig`, `make_batch_sampler`, `train`, `estimate_loss`, `sample`, `cosine_lr`.

## Concepts covered

- The five engineering dials that decide if a training run succeeds (LR schedule, warmup, clip, AdamW betas, bfloat16)
- Cosine learning-rate schedule with linear warmup
- Cross-entropy + autoregressive shift in PyTorch
- TinyStories as a teaching corpus
- Reading a loss curve (spikes, plateaus, divergence)
- Chinchilla scaling and why we deliberately undertrain
- Frontier comparison table (params, tokens, compute, tokens/param)

## Reuses & is reused by

- **Reuses:** `src/model.py` (Post 7) and the TinyStories BPE tokenizer (Post 2).
- **Reused by:** Post 9 loads the checkpoint to demo the KV-cache. Post 11 fine-tunes it with SFT + DPO.
