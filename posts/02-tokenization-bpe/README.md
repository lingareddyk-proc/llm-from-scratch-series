# Post 2 — Tokenization & BPE From Scratch

**Series:** LLM From Scratch · Post **2 of 12** · *Act I — Foundations*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-handle>/llm-from-scratch-series/blob/main/posts/02-tokenization-bpe/notebook.ipynb)

## 60-second summary

Tokenization is how text becomes integers — and the algorithm of choice is **Byte-Pair Encoding (BPE)**, used by GPT-2, GPT-4, Llama 3, and (in spirit) Claude and Gemini. This post implements BPE in ~80 lines of pure Python on a toy corpus, then trains a real **8k-vocab byte-level BPE tokenizer on TinyStories** (which Post 8 will reuse for pretraining). The final cells compare four tokenizers (ours, GPT-2, GPT-4 `cl100k_base`, Llama 3) on English / code / Hindi / emoji to show why your LLM bill scales with tokens, not characters — and why non-English text can cost 2–5× more.

## What's in this folder

| File                                       | What it is                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                 | Medium-ready article.                                                                       |
| [`notebook.ipynb`](notebook.ipynb)         | Colab notebook — 24 cells, runs in ~10 min on free Colab CPU.                               |
| [`_build_notebook.py`](_build_notebook.py) | Source of truth for the notebook. Re-run to regenerate `notebook.ipynb` after edits.        |
| `figures/`                                 | Generated plots (currently empty — plots are produced inside the notebook at runtime).      |

## Running the notebook

**Colab (recommended):** click the badge above.

**Locally:**

```bash
cd posts/02-tokenization-bpe
pip install -r ../../requirements.txt
pip install tiktoken
jupyter lab notebook.ipynb
```

## Concepts covered

- Why not characters? Why not words?
- BPE training loop: count pairs → merge most frequent → repeat
- Byte-level BPE (GPT-2's trick)
- Vocab size trade-offs (training cost vs sequence length vs multilingual coverage)
- Production tokenizers with Hugging Face `tokenizers`
- Tokens-per-character across languages and content types
- Why "128k context" means very different things in English vs Hindi vs code

## Reuses & is reused by

- This notebook **saves** `tinystories_tokenizer/tokenizer.json` at the end.
- **Post 8 (Pretraining)** loads that file and uses it to tokenize the training corpus.
