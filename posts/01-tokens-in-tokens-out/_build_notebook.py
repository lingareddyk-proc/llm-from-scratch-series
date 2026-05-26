"""Generate notebook.ipynb for Post 1.

Run this from the post folder:

    python _build_notebook.py

It writes ./notebook.ipynb. Re-run any time the cells change.
This script is checked in so the notebook is reproducible and reviewable.
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
        """# Post 1 — Tokens In, Tokens Out

Companion notebook to the Medium post. Runs in ~5 min on free Colab CPU. No GPU required.

By the end of this notebook you will have:

- Loaded GPT-2 and inspected its tokenizer
- Run one forward pass and looked at the raw logits
- Built an autoregressive decoding loop by hand
- Visualized the model's top-10 candidates at every step
- Felt the effect of `temperature` and `top-k` on generation

> If you're in **Google Colab**, just run the cells top-to-bottom. Locally, make sure you have a Python 3.10+ env with `torch`, `transformers`, and `matplotlib` installed (or `pip install -r requirements.txt` from the repo root)."""
    ),
    md("## 0. Setup"),
    code('!pip -q install "transformers>=4.44" "torch>=2.3" matplotlib'),
    code(
        """import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)

MODEL_NAME = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

print(f"Loaded {MODEL_NAME}: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Vocab size: {tokenizer.vocab_size:,}")"""
    ),
    md(
        """## 1. Tokens, not words

Let's see what the tokenizer does to a sentence. Notice the leading-space tokens — they are different tokens from the no-space versions."""
    ),
    code(
        """prompt = "The capital of France is"
ids = tokenizer.encode(prompt)
pieces = [tokenizer.decode([i]) for i in ids]

print(f"Prompt:    {prompt!r}")
print(f"Token ids: {ids}")
print("Tokens (each shown with quotes to reveal whitespace):")
for i, p in zip(ids, pieces):
    print(f"  {i:>6}  -> {p!r}")"""
    ),
    md(
        """## 2. One forward pass

Feed the token ids to the model. We get back **logits** of shape `(batch, seq_len, vocab_size)`. For generation we only care about the last position — that's the model's prediction for the **next** token."""
    ),
    code(
        """input_ids = torch.tensor([ids])
with torch.no_grad():
    out = model(input_ids)

logits = out.logits
print(f"logits shape: {tuple(logits.shape)}  # (batch, seq_len, vocab_size)")

next_token_logits = logits[0, -1]
print(f"next-token logits shape: {tuple(next_token_logits.shape)}")
print(f"min={next_token_logits.min():.2f}  max={next_token_logits.max():.2f}")"""
    ),
    md(
        """## 3. From logits to a distribution: softmax

Logits are unnormalized scores. Softmax turns them into probabilities that sum to 1. Let's look at the model's top-10 predictions for what comes after *"The capital of France is"*."""
    ),
    code(
        """def top_k_table(next_token_logits, k=10, temperature=1.0):
    probs = F.softmax(next_token_logits / temperature, dim=-1)
    top = torch.topk(probs, k)
    return [(tokenizer.decode([idx.item()]), p.item()) for p, idx in zip(top.values, top.indices)]

top10 = top_k_table(next_token_logits, k=10)
print(f"{'token':>20}  probability")
print("-" * 40)
for tok, p in top10:
    print(f"{tok!r:>20}  {p:.4f}")"""
    ),
    md(
        """## 4. Visualize it

A bar chart makes the shape of the distribution obvious."""
    ),
    code(
        """def plot_top_k(next_token_logits, k=10, temperature=1.0, title=None):
    rows = top_k_table(next_token_logits, k=k, temperature=temperature)
    tokens = [repr(t) for t, _ in rows]
    probs = [p for _, p in rows]
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(rows))[::-1], probs)
    plt.yticks(range(len(rows))[::-1], tokens)
    plt.xlabel("probability")
    plt.title(title or f"Top {k} candidates (T={temperature})")
    plt.tight_layout()
    plt.show()

plot_top_k(
    next_token_logits,
    k=10,
    temperature=1.0,
    title='Top 10 next tokens after "The capital of France is"',
)"""
    ),
    md(
        """## 5. Temperature

Same logits, different temperatures. `T < 1` sharpens (greedier), `T > 1` flattens (more random)."""
    ),
    code(
        """for T in [0.5, 1.0, 1.5]:
    plot_top_k(next_token_logits, k=10, temperature=T, title=f"T = {T}")"""
    ),
    md(
        """## 6. The autoregressive loop

Now we wrap forward-pass-then-sample into a loop. Append the new token, run the model again, repeat. **This is what every LLM does to generate text.**"""
    ),
    code(
        """def generate(
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int | None = 50,
    verbose: bool = False,
) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids])

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < v[-1], torch.full_like(logits, -float("inf")), logits)

        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if verbose:
            chosen = tokenizer.decode([next_id])
            print(f"step {step:>2}: picked {chosen!r}  (p={probs[next_id].item():.3f})")

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


torch.manual_seed(42)
print(
    generate(
        "The capital of France is",
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        verbose=True,
    )
)"""
    ),
    md(
        """## 7. Watch the model "think" at every step

Same loop, but at every step we draw the top-10 distribution **before** sampling. You're seeing exactly what the model considered."""
    ),
    code(
        """def generate_with_plots(prompt: str, n_steps: int = 5, temperature: float = 1.0):
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids])

    for step in range(n_steps):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]

        so_far = tokenizer.decode(input_ids[0].tolist())
        plot_top_k(
            logits,
            k=10,
            temperature=temperature,
            title=f"step {step} — context: {so_far!r}",
        )

        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    print("\\nFinal text:", tokenizer.decode(input_ids[0].tolist()))


torch.manual_seed(7)
generate_with_plots("Once upon a time, in a quiet village,", n_steps=5, temperature=1.0)"""
    ),
    md(
        """## 8. Greedy vs sampled vs hot

Same prompt, three temperatures. Notice how very low `T` keeps repeating itself (greedy is boring), `T=1.0` is coherent, and `T=2.0` starts to break."""
    ),
    code(
        """prompt = "In the year 2050, scientists discovered that"

for T in [0.01, 1.0, 2.0]:
    torch.manual_seed(0)
    print(f"--- T = {T} ---")
    print(generate(prompt, max_new_tokens=30, temperature=max(T, 1e-4), top_k=50))
    print()"""
    ),
    md(
        """## 9. Cross-entropy loss in one line

During training the model isn't sampling — it's being told the right answer and scored by how surprised it was. Let's compute that loss on a real sentence and see what "low loss" actually feels like."""
    ),
    code(
        """text = "The capital of France is Paris."
ids = tokenizer.encode(text)
input_ids = torch.tensor([ids])

with torch.no_grad():
    logits = model(input_ids).logits

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="none",
)

print(f"{'predicting':>12}  {'given context':<40}  loss")
print("-" * 75)
for i, tok_loss in enumerate(loss):
    target = tokenizer.decode([shift_labels[0, i].item()])
    context = tokenizer.decode(shift_labels[0, :i].tolist())
    print(f"{target!r:>12}  {context!r:<40}  {tok_loss.item():.3f}")

print(f"\\nMean loss:  {loss.mean().item():.3f}")
print(f"Perplexity: {loss.mean().exp().item():.2f}")"""
    ),
    md(
        """## 10. What you just did

You ran the same loop that powers ChatGPT, Claude, and Gemini — just with a 124M-parameter toy model. The architectural ideas are identical; the difference is **scale** (params, data, compute), **engineering** (KV caches, MoE, distributed training), and **post-training** (instruction tuning, RLHF).

Over the next 11 posts we'll progressively open the GPT-2 "black box" we used here, replace it with a modern Llama/Gemma-style decoder we build ourselves, and train it from scratch on TinyStories.

**Next: Post 2 — Tokenization & BPE from scratch.**"""
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
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {
            "provenance": [],
            "toc_visible": True,
        },
    }
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT.relative_to(Path.cwd().parent) if Path.cwd() in OUT.parents else OUT}")


if __name__ == "__main__":
    main()
