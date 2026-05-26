"""Generate notebook.ipynb for Post 2 — Tokenization & BPE from scratch.

Run from this folder:

    python _build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "notebook.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


CELLS: list[nbf.NotebookNode] = [
    md(
        """# Post 2 — Tokenization & BPE From Scratch

Companion notebook to the Medium post. Runs in ~10 min on free Colab CPU.

By the end of this notebook you will:

1. Implement **Byte-Pair Encoding** in ~80 lines of pure Python and watch every merge happen on a toy corpus.
2. Train a production-grade tokenizer with Hugging Face `tokenizers` on **TinyStories**.
3. Compare your tokenizer against **GPT-2**, **GPT-4 / `cl100k_base`**, and **Llama 3** on the same text.
4. Measure **tokens-per-word across English / code / Hindi / emoji** and see why your LLM bill explodes on non-Latin scripts."""
    ),
    md("## 0. Setup"),
    code(
        '!pip -q install "transformers>=4.44" "tokenizers>=0.19" "datasets>=2.20" "tiktoken>=0.7" matplotlib'
    ),
    code(
        """from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt"""
    ),
    md(
        """## 1. BPE from scratch — the entire algorithm in ~80 lines

We'll skip bytes for clarity and work on visible characters with a leading `▁` for word boundaries (this is exactly what SentencePiece does). The algorithm is identical at the byte level."""
    ),
    code(
        '''SPACE = "\u2581"  # the word-boundary marker, visible in printouts


def pretokenize(text: str) -> list[list[str]]:
    """Split text into words, then each word into a list of single chars
    with a leading ▁ to mark the word boundary."""
    words = text.strip().split()
    return [[SPACE + w[0]] + list(w[1:]) if w else [] for w in words]


def get_pair_counts(words: list[list[str]]) -> Counter:
    """Count every adjacent symbol pair across all words."""
    counts: Counter = Counter()
    for word in words:
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += 1
    return counts


def merge_pair(words: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    """Replace every occurrence of `pair` with a single merged symbol."""
    a, b = pair
    merged_symbol = a + b
    new_words = []
    for word in words:
        i = 0
        out = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(merged_symbol)
                i += 2
            else:
                out.append(word[i])
                i += 1
        new_words.append(out)
    return new_words


def train_bpe(text: str, num_merges: int, verbose: bool = False):
    words = pretokenize(text)
    merges: list[tuple[str, str]] = []
    for step in range(num_merges):
        pairs = get_pair_counts(words)
        if not pairs:
            break
        best_pair, best_count = pairs.most_common(1)[0]
        if best_count < 2:
            break
        merges.append(best_pair)
        words = merge_pair(words, best_pair)
        if verbose:
            print(f"merge {step + 1:>3}: {best_pair[0]!r} + {best_pair[1]!r} -> {best_pair[0] + best_pair[1]!r}  (count={best_count})")
    return merges, words


def encode_with_merges(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a single word by re-applying merges in training order."""
    symbols = [SPACE + word[0]] + list(word[1:]) if word else []
    for a, b in merges:
        i = 0
        out = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols'''
    ),
    md(
        """## 2. Train it on a tiny corpus and watch every merge

Classic teaching corpus. Notice how the most common pairs get merged first, then larger and larger chunks form."""
    ),
    code(
        '''toy_corpus = "low low low lower lower newer newer newer newer wider wider wider"
print("Corpus:", toy_corpus)
print()
merges, final_words = train_bpe(toy_corpus, num_merges=15, verbose=True)
print()
print("Final tokenization:")
for w in final_words:
    print("  ", w)'''
    ),
    md(
        """## 3. Encode a new word the tokenizer has never seen

This is the magic of BPE: even unseen words decompose into known sub-pieces — no `<unk>` token needed."""
    ),
    code(
        '''for w in ["low", "lower", "newest", "widely", "supercalifragilistic"]:
    print(f"{w:>22}  ->  {encode_with_merges(w, merges)}")'''
    ),
    md(
        """## 4. Now do it for real: train on TinyStories with the `tokenizers` library

`tokenizers` is the same Rust library that powers Hugging Face's production tokenizers — orders of magnitude faster than our pure-Python version, and supports byte-level BPE, special tokens, padding, truncation, etc.

We'll train a small **8k-vocab** byte-level BPE on a slice of **TinyStories** (the corpus we'll pretrain on in Post 8)."""
    ),
    code(
        '''from datasets import load_dataset

print("Downloading a slice of TinyStories...")
ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
print(f"Loaded {len(ds):,} stories. Example:")
print(ds[0]["text"][:300], "...")'''
    ),
    code(
        '''from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

VOCAB_SIZE = 8_000

tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = __import__("tokenizers").decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True,
)

def iter_corpus():
    for row in ds:
        yield row["text"]

tokenizer.train_from_iterator(iter_corpus(), trainer=trainer, length=len(ds))

print(f"\\nTrained! Final vocab size: {tokenizer.get_vocab_size():,}")'''
    ),
    md("## 5. Encode and decode with our trained tokenizer"),
    code(
        '''sample = "Once upon a time, there was a little girl who loved to read books about dragons."
enc = tokenizer.encode(sample)

print(f"Text:   {sample!r}")
print(f"Tokens: {len(enc.ids)}")
print(f"IDs:    {enc.ids}")
print(f"Pieces: {enc.tokens}")
print(f"Decoded back: {tokenizer.decode(enc.ids)!r}")'''
    ),
    md(
        """## 6. Compare against GPT-2, GPT-4, and Llama 3

We'll tokenize the same passage with four tokenizers and compare lengths.

- **Ours** — 8k vocab, trained on 5k TinyStories rows
- **GPT-2** — 50k vocab (HuggingFace)
- **GPT-4 / cl100k_base** — 100k vocab (`tiktoken`)
- **Llama 3** — 128k vocab (HuggingFace, gated — falls back gracefully if you don't have access)"""
    ),
    code(
        '''import tiktoken
from transformers import AutoTokenizer

passage = (
    "Large language models are trained on trillions of tokens of text. "
    "Each token is a chunk of one to a few characters, produced by a tokenizer. "
    "The choice of tokenizer affects both training efficiency and inference cost."
)

# Our tokenizer
our_n = len(tokenizer.encode(passage).ids)

# GPT-2
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
gpt2_n = len(gpt2_tok.encode(passage))

# GPT-4 (cl100k_base) via tiktoken
cl100k = tiktoken.get_encoding("cl100k_base")
cl100k_n = len(cl100k.encode(passage))

# Llama 3 (open weights, no gating needed for the tokenizer of this fork)
try:
    llama_tok = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
    llama_n = len(llama_tok.encode(passage))
except Exception as e:
    llama_n = None
    print(f"(Llama 3 tokenizer unavailable: {type(e).__name__})")

print(f"Passage: {len(passage)} characters\\n")
print(f"{'tokenizer':<24}{'vocab':>10}{'tokens':>10}{'chars/token':>14}")
print("-" * 58)
for name, vocab, n in [
    ("ours (TinyStories 8k)", 8_000, our_n),
    ("GPT-2", 50_257, gpt2_n),
    ("GPT-4 / cl100k_base", 100_277, cl100k_n),
] + ([("Llama 3", 128_256, llama_n)] if llama_n else []):
    print(f"{name:<24}{vocab:>10,}{n:>10}{len(passage)/n:>14.2f}")'''
    ),
    md(
        """## 7. Tokens per word across languages and content types

The reason a Hindi-speaking customer might cost an LLM provider 3x more than an English-speaking one for the same conversation."""
    ),
    code(
        '''samples = {
    "English prose":  "The quick brown fox jumps over the lazy dog. The cat sat on the mat.",
    "Python code":    "def factorial(n):\\n    if n <= 1: return 1\\n    return n * factorial(n - 1)\\n\\nprint(factorial(10))",
    "Hindi":          "\u092c\u093f\u0932\u094d\u0932\u0940 \u091a\u091f\u093e\u0908 \u092a\u0930 \u092c\u0948\u0920\u0940 \u0925\u0940\u0964 \u092c\u093e\u0939\u0930 \u092c\u093e\u0930\u093f\u0936 \u0939\u094b \u0930\u0939\u0940 \u0925\u0940 \u0914\u0930 \u0939\u0935\u093e \u0924\u0947\u091c \u091a\u0932 \u0930\u0939\u0940 \u0925\u0940\u0964",
    "Emoji":          "\U0001f389\U0001f680\U0001f916\u2728\U0001f525\U0001f4a1\u2764\ufe0f\U0001f1ee\U0001f1f3\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5",
    "Mixed code+English": "// Calculate the area of a circle\\ndef area(r): return 3.14159 * r * r",
}


def count_tokens_all(text: str) -> dict[str, int]:
    out = {
        "ours": len(tokenizer.encode(text).ids),
        "GPT-2": len(gpt2_tok.encode(text)),
        "GPT-4": len(cl100k.encode(text)),
    }
    if llama_n is not None:
        out["Llama 3"] = len(llama_tok.encode(text))
    return out


print(f"{'sample':<22}{'chars':>8}  " + "  ".join(f"{k:>10}" for k in count_tokens_all("a")))
print("-" * 78)
results = {}
for name, text in samples.items():
    counts = count_tokens_all(text)
    results[name] = counts
    print(f"{name:<22}{len(text):>8}  " + "  ".join(f"{v:>10}" for v in counts.values()))'''
    ),
    md(
        """## 8. Visualize the multilingual gap"""
    ),
    code(
        '''import numpy as np

tokenizer_names = list(next(iter(results.values())).keys())
sample_names = list(results.keys())

# tokens per 100 characters — normalizes for length
data = np.array([
    [results[s][t] / len(samples[s]) * 100 for t in tokenizer_names]
    for s in sample_names
])

x = np.arange(len(sample_names))
width = 0.8 / len(tokenizer_names)

plt.figure(figsize=(10, 5))
for i, t in enumerate(tokenizer_names):
    plt.bar(x + i * width, data[:, i], width, label=t)
plt.xticks(x + width * (len(tokenizer_names) - 1) / 2, sample_names, rotation=20, ha="right")
plt.ylabel("tokens per 100 characters\\n(lower = more efficient)")
plt.title("Tokenization efficiency across content types")
plt.legend()
plt.tight_layout()
plt.show()'''
    ),
    md(
        """## 9. The cost implication

If GPT-4 charges per token, then the cost of conveying the same *information* depends on:

- The language (English is cheapest because the tokenizers were trained mostly on English).
- The content type (code is expensive because every space, bracket, and identifier costs tokens).
- The specific tokenizer (newer, larger vocabs help especially on non-English).

This is why frontier labs have been **growing vocabularies** with each generation — Llama 1→2 stayed at 32k, but Llama 3 jumped to 128k, mostly to improve non-English efficiency. Gemini's reported 256k vocab is the same logic taken further."""
    ),
    md(
        """## 10. What you just built

- A working BPE tokenizer in pure Python, complete with merge rules and roundtripping.
- A real 8k-vocab byte-level BPE tokenizer trained on TinyStories — the one we'll reuse to pretrain a model in **Post 8**.
- Quantitative evidence for why tokenization is one of the most under-appreciated parts of LLM design.

**Next: Post 3 — Embeddings and the residual stream.** Those token IDs become vectors. Those vectors live in one shared scratchpad. Everything else — attention, MLPs, MoE — just reads from and writes to it."""
    ),
    code(
        '''# Save our tokenizer so Post 8 can reuse it later.
import os, json
os.makedirs("tinystories_tokenizer", exist_ok=True)
tokenizer.save("tinystories_tokenizer/tokenizer.json")
print("Saved to tinystories_tokenizer/tokenizer.json")
print(f"Vocab size: {tokenizer.get_vocab_size():,}")'''
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
