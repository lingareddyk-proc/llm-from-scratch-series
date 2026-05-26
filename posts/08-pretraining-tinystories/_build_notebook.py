"""Generate notebook.ipynb for Post 8 — Pretraining on TinyStories."""

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
        """# Post 8 — Pretraining a Small LLM End-to-End on Colab

Companion notebook to the Medium post. **Requires a free Colab T4 GPU** (Runtime > Change runtime type > GPU). Total run time ~30 min.

We will:

1. Clone the series repo and load `src/model.py` (`TinyLLM`) and `src/train.py`.
2. Train (or reuse) the 8k-vocab BPE tokenizer from Post 2 on a slice of TinyStories.
3. Tokenize ~13M tokens into one flat tensor.
4. Train a ~6.6M-parameter `TinyLLM` for 2000 steps with AdamW + cosine LR + grad clip + bfloat16.
5. Sample from checkpoints to **watch the model learn** \u2014 gibberish \u2192 syntax \u2192 narrative.
6. Save the trained weights for Posts 9 (KV-cache) and 11 (SFT)."""
    ),
    md("## 0. Setup"),
    code(
        '''!pip -q install "torch>=2.3" "transformers>=4.44" "tokenizers>=0.19" "datasets>=2.20" matplotlib

import os, sys, subprocess, time
REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("ok")'''
    ),
    code(
        '''import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("\\nWarning: training on CPU will be very slow. Switch to a free T4 GPU runtime!")'''
    ),
    md(
        """## 1. Tokenizer \u2014 train (or reuse) the Post 2 BPE on TinyStories"""
    ),
    code(
        '''from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import tokenizers.decoders as td
from datasets import load_dataset

TOK_PATH = "/content/tinystories_tokenizer.json"

if os.path.exists(TOK_PATH):
    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"Loaded existing tokenizer, vocab={tokenizer.get_vocab_size():,}")
else:
    print("Training new 8k BPE tokenizer on 10k TinyStories rows...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:10000]")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = td.ByteLevel()
    trainer = BpeTrainer(vocab_size=8000, special_tokens=["<|endoftext|>", "<|pad|>"], initial_alphabet=ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator((r["text"] for r in ds), trainer=trainer, length=len(ds))
    tokenizer.save(TOK_PATH)
    print(f"Trained, vocab={tokenizer.get_vocab_size():,}")

VOCAB_SIZE = tokenizer.get_vocab_size()
EOS_ID = tokenizer.token_to_id("<|endoftext|>")'''
    ),
    md(
        """## 2. Tokenize a TinyStories slice \u2014 ~13M tokens

We append the EOS token between stories so the model learns story boundaries."""
    ),
    code(
        '''N_STORIES = 40_000  # ~13M tokens after encoding; tweak down to 10k if you're impatient
print(f"Loading {N_STORIES:,} TinyStories rows...")
ds = load_dataset("roneneldan/TinyStories", split=f"train[:{N_STORIES}]")

t0 = time.time()
all_ids: list[int] = []
for i, row in enumerate(ds):
    ids = tokenizer.encode(row["text"]).ids
    all_ids.extend(ids)
    all_ids.append(EOS_ID)
    if (i + 1) % 10000 == 0:
        print(f"  {i+1:,} stories, {len(all_ids):,} tokens, {(time.time()-t0):.1f}s")

token_ids = torch.tensor(all_ids, dtype=torch.long)
n_train = int(0.99 * token_ids.numel())
train_ids = token_ids[:n_train]
val_ids   = token_ids[n_train:]
print(f"\\ntotal tokens: {token_ids.numel():,}")
print(f"train: {train_ids.numel():,}  val: {val_ids.numel():,}")'''
    ),
    md(
        """## 3. Build the model from `src/model.py`"""
    ),
    code(
        '''from src.model import ModelConfig, TinyLLM

cfg = ModelConfig(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_layers=6,
    n_heads_q=8,
    n_heads_kv=2,
    d_ff=768,
    max_seq_len=256,
)
print(cfg)

model = TinyLLM(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\\nmodel params: {n_params:,}  ({n_params/1e6:.2f} M)")'''
    ),
    md(
        """## 4. Train

The whole loop \u2014 sampler, AdamW, cosine LR, grad clip, bfloat16 \u2014 lives in `src/train.py`. ~120 lines, read it.

This trains for **2000 steps**. On a T4 expect roughly **30 minutes**. If you want a quicker smoke run, drop `total_steps` to 500 (you'll see syntax emerge but not narrative)."""
    ),
    code(
        '''from src.train import TrainConfig, make_batch_sampler, train, sample

train_sampler = make_batch_sampler(train_ids, batch_size=32, seq_len=cfg.max_seq_len, device=device)
val_sampler   = make_batch_sampler(val_ids,   batch_size=32, seq_len=cfg.max_seq_len, device=device)

tcfg = TrainConfig(
    batch_size=32,
    seq_len=cfg.max_seq_len,
    lr_max=3e-4,
    lr_min=3e-5,
    warmup_steps=100,
    total_steps=2000,
    weight_decay=0.1,
    grad_clip=1.0,
    log_every=50,
    eval_every=250,
    autocast_dtype=torch.bfloat16,
)

prompt_ids = torch.tensor([tokenizer.encode("Once upon a time").ids], device=device)
snapshots: list[tuple[int, str]] = []

def on_eval(step, train_loss, val_loss):
    out = sample(model, prompt_ids, max_new_tokens=60, temperature=0.8, top_k=100)
    text = tokenizer.decode(out[0].tolist())
    snapshots.append((step, text))
    print(f"  >> sample @ step {step}: {text!r}")

history = train(model, train_sampler, val_sampler, tcfg, on_eval=on_eval)'''
    ),
    md(
        """## 5. Plot the loss curve"""
    ),
    code(
        '''import matplotlib.pyplot as plt

t_steps, t_losses = zip(*history["train"])
v_steps, v_losses = zip(*history["val"])

plt.figure(figsize=(9, 5))
plt.plot(t_steps, t_losses, label="train", alpha=0.7)
plt.plot(v_steps, v_losses, label="val", linewidth=2, marker='o')
plt.xlabel("step"); plt.ylabel("cross-entropy loss")
plt.title("TinyLLM pretraining on TinyStories")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 6. Watch the model learn

Side-by-side: what the model generated at each evaluation checkpoint, given the same prompt."""
    ),
    code(
        '''print(f"Prompt: 'Once upon a time'\\n")
print("=" * 80)
for step, text in snapshots:
    print(f"\\n[step {step}]")
    print(text)
    print("-" * 80)'''
    ),
    md(
        """## 7. Sample with different temperatures"""
    ),
    code(
        '''prompt = "Once upon a time, in a quiet village,"
prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)

for T in [0.4, 0.8, 1.2]:
    out = sample(model, prompt_ids, max_new_tokens=80, temperature=T, top_k=100)
    print(f"--- T={T} ---")
    print(tokenizer.decode(out[0].tolist()))
    print()'''
    ),
    md(
        """## 8. Save the checkpoint

Posts 9 (KV-cache) and 11 (SFT) load this file. Save the tokenizer alongside it."""
    ),
    code(
        '''CKPT_PATH = "/content/tinyllm_pretrained.pt"

torch.save({
    "model_state": model.state_dict(),
    "config": cfg.__dict__,
}, CKPT_PATH)
print(f"Saved {CKPT_PATH} ({os.path.getsize(CKPT_PATH)/1e6:.1f} MB)")

import shutil
shutil.copy(TOK_PATH, "/content/tinystories_tokenizer_for_post9.json")
print("Tokenizer copied for downstream posts.")'''
    ),
    md(
        """## 9. What you just did

- Trained a real (small) modern Llama-style LLM from random init, on real text, in 30 minutes on a free GPU.
- Watched cross-entropy loss drop from ~9.0 to ~3.0.
- Watched the model itself go from gibberish to coherent short stories.

You've now done the same thing OpenAI, Anthropic, Google, and Meta do \u2014 just at 6 orders of magnitude smaller scale.

**Next: Post 9 \u2014 Sampling and the KV-cache.** Generation right now is slow because every new token recomputes attention over the whole prefix. We'll fix that and benchmark the speedup."""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    }
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
