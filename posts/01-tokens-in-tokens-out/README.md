# Post 1 — Tokens In, Tokens Out

**Series:** LLM From Scratch · Post **1 of 12** · *Act I — Foundations*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-handle>/llm-from-scratch-series/blob/main/posts/01-tokens-in-tokens-out/notebook.ipynb)

## 60-second summary

Every LLM — from GPT-2 to Gemini 3 — runs the same outer loop: feed text in, get a probability distribution over the next token, sample one, append, repeat. This post implements that loop by hand against pretrained GPT-2 in ~30 lines of PyTorch, visualizes the model's top-10 candidates at every step, and shows how `temperature` and `top-k` shape the output. The final cell computes cross-entropy loss — the one number every frontier LLM is optimizing during pretraining.

## What's in this folder

| File                                          | What it is                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [`ARTICLE.md`](ARTICLE.md)                    | Medium-ready article. Paste straight into the Medium editor.                                 |
| [`notebook.ipynb`](notebook.ipynb)            | Companion Colab notebook. 23 cells, runs in ~5 min on free Colab CPU.                        |
| [`_build_notebook.py`](_build_notebook.py)    | Source of truth for the notebook. Re-run to regenerate `notebook.ipynb` after edits.         |
| `figures/`                                    | Generated plots (currently empty — all figures are produced inside the notebook at runtime). |

## Running the notebook

**Colab (recommended):** click the badge above.

**Locally:**

```bash
cd posts/01-tokens-in-tokens-out
pip install -r ../../requirements.txt
jupyter lab notebook.ipynb
```

## Concepts covered

- Autoregressive decoding
- Tokens vs words (preview of Post 2)
- Logits, softmax, temperature, top-k sampling
- Cross-entropy loss & perplexity
- How the same loop scales to frontier models (Gemini, Claude, GPT-4)
