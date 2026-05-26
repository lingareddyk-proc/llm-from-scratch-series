"""Generate notebook.ipynb for Post 11 — SFT and DPO."""

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
        """# Post 11 \u2014 SFT and DPO

Companion notebook to the Medium post. ~10 min, free Colab CPU or T4.

We will:

1. Load the Post 8 base model (or train a fresh small one if missing).
2. **SFT** it on a tiny instruction dataset with masked cross-entropy.
3. Compare base vs SFT responses on the same prompts.
4. Run **DPO** on a handful of preference pairs.
5. Show that the chosen-vs-rejected likelihood ratio moves in the right direction."""
    ),
    md("## 0. Setup"),
    code(
        '''!pip -q install "torch>=2.3" "tokenizers>=0.19" "datasets>=2.20" matplotlib

import os, sys, subprocess, copy
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)'''
    ),
    md(
        """## 1. Load base model and tokenizer (or initialize fresh)"""
    ),
    code(
        '''from src.model import ModelConfig, TinyLLM
from tokenizers import Tokenizer

CKPT_PATH = "/content/tinyllm_pretrained.pt"
TOK_PATH  = "/content/tinystories_tokenizer.json"

if os.path.exists(CKPT_PATH) and os.path.exists(TOK_PATH):
    print("Loading Post 8 checkpoint and tokenizer.")
    obj = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ModelConfig(**obj["config"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(obj["model_state"])
    tokenizer = Tokenizer.from_file(TOK_PATH)
else:
    print("Initialising fresh model for demo \u2014 fine-tuning math is the same.")
    cfg = ModelConfig(vocab_size=8000, d_model=192, n_layers=4, n_heads_q=6, n_heads_kv=2, d_ff=512, max_seq_len=256)
    model = TinyLLM(cfg).to(device)
    tokenizer = None

print(f"params: {sum(p.numel() for p in model.parameters()):,}")'''
    ),
    md(
        """## 2. Build a tiny instruction dataset

Since our pretrained model is only ~6M parameters trained on TinyStories, real instruction-following data won't help much. We'll use a **deterministic toy task**: "Say hello in <language>". A model this small can actually learn this."""
    ),
    code(
        '''languages = {
    "English": "Hello.", "French": "Bonjour.", "Spanish": "Hola.",
    "German": "Hallo.", "Italian": "Ciao.", "Portuguese": "Ola.",
    "Dutch": "Hallo.", "Norwegian": "Hei.", "Swedish": "Hej.",
}

USER_TAG = "USER: "
ASSISTANT_TAG = "\\nASSISTANT: "
END = "\\n###\\n"

def make_pair(lang, greeting):
    user = f"Say hello in {lang}."
    asst = greeting
    prompt = f"{USER_TAG}{user}{ASSISTANT_TAG}"
    full = prompt + asst + END
    return prompt, full

import random
random.seed(0)
pairs = []
for _ in range(200):
    lang, greeting = random.choice(list(languages.items()))
    pairs.append(make_pair(lang, greeting))

print(pairs[0][1])
print("---")
print(pairs[1][1])'''
    ),
    md(
        """## 3. SFT \u2014 cross-entropy with the prompt masked out

We tokenize each `(prompt, full)` pair, set the loss target to `-100` for every prompt token, and only train on the assistant tokens."""
    ),
    code(
        '''IGN = -100

def tokenize_pair(prompt, full):
    p_ids = tokenizer.encode(prompt).ids if tokenizer else list(range(len(prompt) % 50))
    f_ids = tokenizer.encode(full).ids   if tokenizer else list(range(len(full)   % 100))
    if len(f_ids) > cfg.max_seq_len:
        f_ids = f_ids[:cfg.max_seq_len]
    labels = list(f_ids)
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = IGN
    return torch.tensor(f_ids[:-1], dtype=torch.long), torch.tensor(labels[1:], dtype=torch.long)

batches = [tokenize_pair(p, f) for p, f in pairs]
# Pad to batches of fixed size
PAD = (tokenizer.token_to_id("<|pad|>") if tokenizer else 0)
def collate(batch, B):
    chunks = [batch[i:i+B] for i in range(0, len(batch), B)]
    out = []
    for ch in chunks:
        max_len = max(x[0].size(0) for x in ch)
        x = torch.full((len(ch), max_len), PAD, dtype=torch.long)
        y = torch.full((len(ch), max_len), IGN, dtype=torch.long)
        for i, (xi, yi) in enumerate(ch):
            x[i, :xi.size(0)] = xi
            y[i, :yi.size(0)] = yi
        out.append((x.to(device), y.to(device)))
    return out

print(f"#training pairs: {len(batches)}")'''
    ),
    code(
        '''import copy
sft_model = copy.deepcopy(model).to(device)
opt = torch.optim.AdamW(sft_model.parameters(), lr=5e-4, weight_decay=0.01)

sft_losses = []
for epoch in range(8):
    bs = collate(batches, B=8)
    random.shuffle(bs)
    for x, y in bs:
        logits = sft_model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGN)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
        opt.step()
        sft_losses.append(loss.item())
    print(f"epoch {epoch+1}/8  loss {sum(sft_losses[-len(bs):])/len(bs):.3f}")

plt.figure(figsize=(8, 3))
plt.plot(sft_losses); plt.xlabel("step"); plt.ylabel("masked CE loss")
plt.title("SFT training loss"); plt.tight_layout(); plt.show()'''
    ),
    md("""## 4. Sample base vs SFT \u2014 same prompts"""),
    code(
        '''@torch.no_grad()
def greedy(m, prompt, max_new=20):
    m.eval()
    if tokenizer is None:
        return "(no tokenizer)"
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    for _ in range(max_new):
        logits = m(ids[:, -cfg.max_seq_len:])[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if "###" in tokenizer.decode(ids[0].tolist())[-6:]:
            break
    return tokenizer.decode(ids[0].tolist())

for lang in ["French", "Spanish", "German"]:
    p = f"USER: Say hello in {lang}.\\nASSISTANT: "
    print(f"--- {lang} ---")
    print("BASE:", greedy(model, p, max_new=15))
    print("SFT :", greedy(sft_model, p, max_new=15))
    print()'''
    ),
    md(
        """## 5. DPO \u2014 prefer chosen over rejected, relative to a frozen reference

For each preference pair `(prompt, chosen, rejected)` compute:

- `log_pi(chosen) - log_pi_ref(chosen)` and likewise for rejected.
- DPO loss: `-log_sigmoid(beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)))`.

Below we use a small hand-crafted preference set where "rejected" answers are wrong-language or off-task."""
    ),
    code(
        '''preferences = [
    ("USER: Say hello in French.\\nASSISTANT: ", "Bonjour.\\n###\\n", "Hola.\\n###\\n"),
    ("USER: Say hello in Spanish.\\nASSISTANT: ", "Hola.\\n###\\n", "Bonjour.\\n###\\n"),
    ("USER: Say hello in German.\\nASSISTANT: ", "Hallo.\\n###\\n", "Hello.\\n###\\n"),
    ("USER: Say hello in Italian.\\nASSISTANT: ", "Ciao.\\n###\\n", "Once upon a time...\\n###\\n"),
] * 8   # x8 to give the optimizer a few steps

def encode(s):
    return torch.tensor(tokenizer.encode(s).ids, dtype=torch.long, device=device)

def sequence_logprob(m, prompt_ids, full_ids):
    """log p(full_ids | prompt_ids) under m \u2014 just the assistant-token logprobs."""
    x = full_ids[:-1].unsqueeze(0)
    y = full_ids[1:].unsqueeze(0)
    logits = m(x)
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    # mask: only count tokens after prompt
    pl = prompt_ids.size(0)
    mask = torch.zeros_like(chosen)
    mask[:, pl - 1:] = 1.0
    return (chosen * mask).sum() / mask.sum().clamp(min=1)

ref_model = copy.deepcopy(sft_model).to(device).eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

beta = 0.1
opt = torch.optim.AdamW([p for p in sft_model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)

dpo_losses, margins = [], []
for step in range(80):
    p_text, chosen, rejected = preferences[step % len(preferences)]
    p_ids   = encode(p_text)
    win_ids = encode(p_text + chosen)
    los_ids = encode(p_text + rejected)

    log_pi_w  = sequence_logprob(sft_model, p_ids, win_ids)
    log_ref_w = sequence_logprob(ref_model, p_ids, win_ids)
    log_pi_l  = sequence_logprob(sft_model, p_ids, los_ids)
    log_ref_l = sequence_logprob(ref_model, p_ids, los_ids)

    logits = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
    loss = -F.logsigmoid(logits)

    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    opt.step()

    dpo_losses.append(loss.item())
    margins.append((log_pi_w - log_pi_l).item())

    if (step + 1) % 20 == 0:
        print(f"step {step+1}: loss {loss.item():.3f}  log p(win)-log p(lose) = {margins[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(dpo_losses); axes[0].set_title("DPO loss"); axes[0].set_xlabel("step")
axes[1].plot(margins); axes[1].set_title("log p(chosen) - log p(rejected)"); axes[1].set_xlabel("step")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 6. Verify the model now prefers the chosen answer"""
    ),
    code(
        '''sft_model.eval()
print(f"{'prompt':<50}{'win':>10}{'lose':>10}{'margin':>10}")
print("-" * 80)
for p_text, win, lose in preferences[:4]:
    p_ids   = encode(p_text)
    win_ids = encode(p_text + win)
    los_ids = encode(p_text + lose)
    lw = sequence_logprob(sft_model, p_ids, win_ids).item()
    ll = sequence_logprob(sft_model, p_ids, los_ids).item()
    print(f"{p_text.replace(chr(10), ' '):<50}{lw:>10.3f}{ll:>10.3f}{lw-ll:>10.3f}")'''
    ),
    md(
        """## 7. What you just did

- Took a base model and **SFT-fine-tuned** it on a tiny instruction dataset using masked cross-entropy.
- Implemented **DPO** \u2014 the modern simpler alternative to RLHF/PPO \u2014 from scratch.
- Verified the SFT+DPO model has a higher likelihood for "good" answers vs "bad" answers, by exactly the mechanism the DPO loss is designed to produce.

You've now walked through the entire frontier-LLM recipe \u2014 pretraining (Post 8), SFT (here), DPO (here) \u2014 in code. Same recipe Claude, Gemini, Llama-Instruct, and Mistral-Instruct use.

**Next: Post 12 \u2014 mapping our whole stack to Gemini 3 / Claude.**"""
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
