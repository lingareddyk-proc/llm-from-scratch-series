"""Generate notebook.ipynb for Post 12 — Putting it together."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "notebook.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


CELLS = [
    md(
        """# Post 12 \u2014 Putting It Together

The capstone notebook. Loads `src/model.py` (TinyLLM), trains briefly or loads the Post 8 checkpoint, generates with the KV-cache, and prints an architecture map vs Llama 3 8B."""
    ),
    md("## 0. Setup"),
    code(
        '''!pip -q install "torch>=2.3" "tokenizers>=0.19" "datasets>=2.20" matplotlib

import os, sys, subprocess, time
import torch, torch.nn as nn, torch.nn.functional as F

REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)'''
    ),
    md("## 1. Build the model"),
    code(
        '''from src.model import ModelConfig, TinyLLM

cfg = ModelConfig(
    vocab_size=8_000,
    d_model=256,
    n_layers=6,
    n_heads_q=8,
    n_heads_kv=2,
    d_ff=768,
    max_seq_len=256,
)
model = TinyLLM(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"TinyLLM parameters: {n_params:,}  ({n_params/1e6:.2f} M)")'''
    ),
    md(
        """## 2. Component inventory \u2014 print every module class with its parameter count

This is the "X-ray view" of a modern LLM. Every name here was introduced in an earlier post."""
    ),
    code(
        '''from collections import defaultdict

def summarize(model):
    rows = []
    for name, mod in model.named_modules():
        children = list(mod.children())
        if children or not list(mod.parameters(recurse=False)):
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        rows.append((name, mod.__class__.__name__, n))
    return rows

rows = summarize(model)
# Aggregate by class
agg = defaultdict(lambda: [0, 0])
for _, cls, n in rows:
    agg[cls][0] += 1
    agg[cls][1] += n

print(f"{'class':<22}{'count':>10}{'params':>15}")
print("-" * 47)
for cls, (count, n) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"{cls:<22}{count:>10}{n:>15,}")
print("-" * 47)
print(f"{'TOTAL':<22}{'':>10}{sum(p.numel() for p in model.parameters()):>15,}")'''
    ),
    md("## 3. Map every layer to its post"),
    code(
        '''legend = {
    "Embedding":            "Post 3  \u2014 embeddings + tied lm_head",
    "RMSNorm":              "Post 7  \u2014 modern decoder block",
    "Linear":               "Posts 4/5/7 \u2014 W_q/W_k/W_v/W_o, SwiGLU matrices",
    "GroupedQueryAttention":"Post 5/7 \u2014 GQA + RoPE",
    "SwiGLU":               "Post 7  \u2014 modern FFN",
    "DecoderBlock":         "Post 7  \u2014 Pre-Norm wrapper",
    "TinyLLM":              "Post 7  \u2014 full assembled model",
}
for cls, where in legend.items():
    print(f"  {cls:<24} -> {where}")'''
    ),
    md(
        """## 4. Quick training run (or skip if you have a Post 8 checkpoint)"""
    ),
    code(
        '''from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import tokenizers.decoders as td

TOK_PATH  = "/content/tinystories_tokenizer.json"
CKPT_PATH = "/content/tinyllm_pretrained.pt"

if not os.path.exists(TOK_PATH):
    print("Training tokenizer...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:3000]")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = td.ByteLevel()
    trainer = BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=["<|endoftext|>", "<|pad|>"], initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator((r["text"] for r in ds), trainer=trainer, length=len(ds))
    tokenizer.save(TOK_PATH)
tokenizer = Tokenizer.from_file(TOK_PATH)
EOS = tokenizer.token_to_id("<|endoftext|>")

if os.path.exists(CKPT_PATH):
    obj = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(obj["model_state"])
    print("Loaded checkpoint from Post 8.")
else:
    print("No checkpoint \u2014 doing a tiny 400-step training run for demo purposes.")
    from src.train import TrainConfig, make_batch_sampler, train
    ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
    all_ids = []
    for r in ds:
        all_ids.extend(tokenizer.encode(r["text"]).ids); all_ids.append(EOS)
    token_ids = torch.tensor(all_ids, dtype=torch.long)
    n_train = int(0.99 * token_ids.numel())
    train_sampler = make_batch_sampler(token_ids[:n_train], 16, cfg.max_seq_len, device)
    val_sampler   = make_batch_sampler(token_ids[n_train:], 16, cfg.max_seq_len, device)
    tcfg = TrainConfig(batch_size=16, seq_len=cfg.max_seq_len, total_steps=400, eval_every=200, warmup_steps=50, log_every=100)
    train(model, train_sampler, val_sampler, tcfg)
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, CKPT_PATH)
    print(f"Saved {CKPT_PATH}")'''
    ),
    md("## 5. Generate with the KV-cache (or naive fallback)"),
    code(
        '''from src.train import sample

prompt = "Once upon a time"
ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
out = sample(model, ids, max_new_tokens=80, temperature=0.8, top_k=80)
print(tokenizer.decode(out[0].tolist()))'''
    ),
    md(
        """## 6. Architecture vs Llama 3 8B \u2014 same shape, different numbers"""
    ),
    code(
        '''comparison = [
    ("vocab_size",     cfg.vocab_size,            128_256),
    ("d_model",        cfg.d_model,                 4_096),
    ("layers",         cfg.n_layers,                   32),
    ("Q heads",        cfg.n_heads_q,                  32),
    ("KV heads (GQA)", cfg.n_heads_kv,                  8),
    ("d_head",         cfg.d_head,                    128),
    ("d_ff (SwiGLU)",  cfg.d_ff,                   14_336),
    ("position",       "RoPE",                     "RoPE"),
    ("norm",           "RMSNorm Pre-Norm",  "RMSNorm Pre-Norm"),
    ("FFN",            "SwiGLU",                   "SwiGLU"),
    ("Total params",   f"{n_params:,}",       "8,030,261,248"),
]

print(f"{'':<18}{'OURS (TinyLLM)':>20}{'LLAMA 3 8B':>22}")
print("-" * 60)
for name, a, b in comparison:
    print(f"{name:<18}{str(a):>20}{str(b):>22}")'''
    ),
    md(
        """## 7. The full series in one diagram"""
    ),
    code(
        '''pipeline = [
    ("Post 1",  "tokens-in-tokens-out",        "Autoregressive decoding loop"),
    ("Post 2",  "tokenization-bpe",            "Byte-Pair Encoding tokenizer"),
    ("Post 3",  "embeddings-residual-stream",  "Embeddings + residual stream"),
    ("Post 4",  "attention-from-first-principles", "Scaled dot-product attention"),
    ("Post 5",  "multi-head-attention",        "Multi-head attention (heads specialize)"),
    ("Post 6",  "positional-rope",             "RoPE (rotary position)"),
    ("Post 7",  "modern-decoder-block",        "Llama/Gemma-style block + TinyLLM"),
    ("Post 8",  "pretraining-tinystories",     "Train from scratch on TinyStories"),
    ("Post 9",  "sampling-kvcache",            "KV-cache and modern sampling"),
    ("Post 10", "mixture-of-experts",          "MoE for frontier-style scaling"),
    ("Post 11", "sft-and-dpo",                 "Turn base model into an assistant"),
    ("Post 12", "frontier-llm-mapping",        "(this) end-to-end recap"),
]

print(f"{'#':<7}{'folder':<32}{'topic'}")
print("-" * 80)
for n, folder, topic in pipeline:
    print(f"{n:<7}{folder:<32}{topic}")'''
    ),
    md(
        """## 8. Thank you

You wrote the architecture of a modern LLM. Same equations, same training loop, same post-training recipe as Gemini and Claude \u2014 just (much, much) smaller. From here:

- **Scale up** with bigger `d_model` and more steps on a paid GPU.
- **Read** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and the Llama 3 / Gemma 2 / DeepSeek V3 papers \u2014 you'll recognise everything.
- **Teach** the series to someone else; that's the best way to lock it in.

\u2014 *Linga Reddy K*"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
