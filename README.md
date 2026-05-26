# LLM From Scratch — a 12-post blog series

A practical, code-first walk through transformer architecture that progressively builds a modern Llama / Gemma-style decoder LLM in Google Colab — and shows how each piece maps to what frontier models (Gemini, Claude, GPT-4) actually do.

Each post in the series ships:

- A **Medium article** (`ARTICLE.md` in the post folder).
- A self-contained **Colab notebook** (`notebook.ipynb`) — runnable in under 15 minutes on a free Colab runtime.
- A short **`README.md`** with an "Open in Colab" badge and a 60-second summary.

The shared Python package under [`src/`](src/) grows post-by-post (attention, model, training utilities). Older posts continue to work because each notebook installs the repo at a pinned commit.

## The arc

```
Act I  — Foundations             (Posts 1–4)
Act II — Build a modern decoder  (Posts 5–8)
Act III — Frontier techniques    (Posts 9–12)
```

| #  | Title                                                                | Folder                                                                                          |
| -- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1  | What an LLM actually is: tokens in, tokens out                       | [`posts/01-tokens-in-tokens-out`](posts/01-tokens-in-tokens-out)                                |
| 2  | Tokenization: how text becomes numbers (BPE from scratch)            | [`posts/02-tokenization-bpe`](posts/02-tokenization-bpe)                                        |
| 3  | Embeddings and the residual stream                                   | [`posts/03-embeddings-residual-stream`](posts/03-embeddings-residual-stream)                    |
| 4  | Attention from first principles                                      | [`posts/04-attention-from-first-principles`](posts/04-attention-from-first-principles)          |
| 5  | Multi-head attention and why heads specialize                        | _coming soon_                                                                                   |
| 6  | Positional information: from sinusoids to RoPE                       | _coming soon_                                                                                   |
| 7  | The full modern decoder block (Llama / Gemma style)                  | _coming soon_                                                                                   |
| 8  | Pretraining a small LLM end-to-end on Colab                          | _coming soon_                                                                                   |
| 9  | Sampling, KV-cache, and making inference fast                        | _coming soon_                                                                                   |
| 10 | Mixture of Experts: how Gemini and GPT-4 scale further               | _coming soon_                                                                                   |
| 11 | From base model to assistant: SFT and DPO                            | _coming soon_                                                                                   |
| 12 | Putting it together: how a frontier LLM is actually built            | _coming soon_                                                                                   |

## Run locally

```bash
git clone https://github.com/lingareddyk-proc/llm-from-scratch-series.git
cd llm-from-scratch-series
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Run on Google Colab

Each post folder has an "Open in Colab" badge in its `README.md` — one click and you're in.

## Layout

```
llm-from-scratch-series/
├── README.md                         <- you are here
├── requirements.txt
├── LICENSE                           <- MIT
├── src/                              <- shared package, grows over the series
│   └── __init__.py
└── posts/
    ├── 01-tokens-in-tokens-out/
    │   ├── ARTICLE.md                <- Medium-ready article
    │   ├── README.md                 <- Colab badge + summary
    │   ├── notebook.ipynb            <- Colab notebook
    │   └── figures/                  <- generated plots, if any
    └── 02-tokenization-bpe/          <- same layout per post
```

## License

MIT — see [`LICENSE`](LICENSE).
