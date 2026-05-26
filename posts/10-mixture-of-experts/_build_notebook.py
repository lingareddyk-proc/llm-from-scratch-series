"""Generate notebook.ipynb for Post 10 — Mixture of Experts."""

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
        """# Post 10 — Mixture of Experts

Companion notebook to the Medium post. ~15 min on free Colab CPU or T4.

We will:

1. Build a small `MoESwiGLU` module \u2014 4 experts, top-1 routing, with load-balance loss.
2. Build an `MoETinyLLM` by swapping our SwiGLU for the MoE version.
3. Train it briefly on TinyStories.
4. **Visualize expert specialization** \u2014 plot which expert each token routes to."""
    ),
    md("## 0. Setup"),
    code(
        '''!pip -q install "torch>=2.3" "tokenizers>=0.19" "datasets>=2.20" matplotlib

import os, sys, subprocess, time
REPO_URL = "https://github.com/lingareddyk-proc/llm-from-scratch-series.git"
REPO_DIR = "/content/llm-from-scratch-series"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)'''
    ),
    md(
        """## 1. The MoE module

A pool of expert SwiGLU FFNs, a `Linear` gate, top-1 routing, plus a load-balance loss."""
    ),
    code(
        '''from src.model import ModelConfig, RMSNorm, SwiGLU


class MoESwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, top_k: int = 1, balance_coef: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_coef = balance_coef
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])
        self.last_aux_loss = torch.tensor(0.0)
        self.last_routing = None  # (B*T,) of chosen expert ids for inspection

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        gate_logits = self.gate(x_flat)                       # (BT, N)

        # top-k gating
        top_logits, top_idx = gate_logits.topk(self.top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)           # (BT, k)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] = out[mask] + top_weights[mask, k:k+1] * self.experts[e](x_flat[mask])

        # Load balancing aux loss (Switch Transformer style)
        gate_probs = F.softmax(gate_logits, dim=-1)            # (BT, N)
        # Fraction of tokens whose top-1 was each expert
        top1 = top_idx[:, 0]
        f = F.one_hot(top1, num_classes=self.n_experts).float().mean(dim=0)   # (N,)
        # Mean gate prob per expert
        P = gate_probs.mean(dim=0)                              # (N,)
        self.last_aux_loss = self.balance_coef * self.n_experts * (f * P).sum()
        self.last_routing = top1.detach().cpu()

        return out.view(B, T, D)


# Quick shape check
torch.manual_seed(0)
moe = MoESwiGLU(d_model=64, d_ff=128, n_experts=4, top_k=1)
x = torch.randn(2, 6, 64)
y = moe(x)
print(f"input  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}")
print(f"aux loss: {moe.last_aux_loss.item():.4f}")
print(f"routing per token: {moe.last_routing.tolist()}")'''
    ),
    md(
        """## 2. An MoE decoder block and full model

Identical to `DecoderBlock`/`TinyLLM` from Post 7 except the FFN is now MoE."""
    ),
    code(
        '''from src.model import GroupedQueryAttention


class MoEDecoderBlock(nn.Module):
    def __init__(self, cfg, n_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = MoESwiGLU(cfg.d_model, cfg.d_ff, n_experts=n_experts, top_k=top_k)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MoETinyLLM(nn.Module):
    def __init__(self, cfg, n_experts=4, top_k=1):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MoEDecoderBlock(cfg, n_experts, top_k) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, ids):
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x) @ self.tok_emb.weight.T

    def aux_loss(self):
        return sum(blk.ffn.last_aux_loss for blk in self.blocks)


cfg = ModelConfig(vocab_size=8000, d_model=192, n_layers=4, n_heads_q=6, n_heads_kv=2, d_ff=512, max_seq_len=256)
moe_model = MoETinyLLM(cfg, n_experts=4, top_k=1).to(device)
n_total = sum(p.numel() for p in moe_model.parameters())
# Active params per token = embedding + non-FFN params per block * n_blocks + 1 expert per FFN
print(f"MoE TinyLLM total params: {n_total:,}")
ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
logits = moe_model(ids)
print(f"logits: {tuple(logits.shape)}, aux loss: {moe_model.aux_loss().item():.4f}")'''
    ),
    md(
        """## 3. Quick training run on TinyStories"""
    ),
    code(
        '''from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import tokenizers.decoders as td

TOK_PATH = "/content/tinystories_tokenizer.json"
if not os.path.exists(TOK_PATH):
    print("Training tokenizer...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = td.ByteLevel()
    trainer = BpeTrainer(vocab_size=8000, special_tokens=["<|endoftext|>", "<|pad|>"], initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator((r["text"] for r in ds), trainer=trainer, length=len(ds))
    tokenizer.save(TOK_PATH)
tokenizer = Tokenizer.from_file(TOK_PATH)
EOS = tokenizer.token_to_id("<|endoftext|>")
print(f"vocab={tokenizer.get_vocab_size():,}")'''
    ),
    code(
        '''ds = load_dataset("roneneldan/TinyStories", split="train[:10000]")
all_ids = []
for r in ds:
    all_ids.extend(tokenizer.encode(r["text"]).ids); all_ids.append(EOS)
token_ids = torch.tensor(all_ids, dtype=torch.long)
print(f"total tokens: {token_ids.numel():,}")

def get_batch(B, T):
    ix = torch.randint(0, token_ids.numel() - T - 1, (B,))
    x = torch.stack([token_ids[i:i+T] for i in ix]).to(device)
    y = torch.stack([token_ids[i+1:i+1+T] for i in ix]).to(device)
    return x, y

opt = torch.optim.AdamW(moe_model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

losses, aux_losses = [], []
for step in range(1, 501):
    x, y = get_batch(B=16, T=128)
    logits = moe_model(x)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    aux = moe_model.aux_loss()
    loss = ce + aux
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 1.0)
    opt.step()
    losses.append(ce.item()); aux_losses.append(aux.item())
    if step % 50 == 0:
        print(f"step {step}: ce {ce.item():.3f}  aux {aux.item():.4f}")'''
    ),
    code(
        '''fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(losses, label="CE loss", color="C0", alpha=0.7); ax1.set_ylabel("CE loss", color="C0")
ax2 = ax1.twinx()
ax2.plot(aux_losses, label="aux loss", color="C1", alpha=0.7); ax2.set_ylabel("aux load-balance loss", color="C1")
ax1.set_xlabel("step"); plt.title("MoETinyLLM training")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 4. Visualize expert routing

For a held-out sentence, dump which expert every token went to in every layer. Look for visible specialization \u2014 a column being mostly one color = an expert that specializes in that token type / position."""
    ),
    code(
        '''sentence = "Once upon a time there was a little girl who loved her cat."
ids = torch.tensor([tokenizer.encode(sentence).ids], device=device)
tokens = [tokenizer.decode([i]).strip() or "_" for i in ids[0].tolist()]

moe_model.eval()
with torch.no_grad():
    _ = moe_model(ids)

L = len(moe_model.blocks)
B = 1
T = ids.size(1)
routing = torch.stack([blk.ffn.last_routing.view(B, T)[0] for blk in moe_model.blocks])  # (L, T)

plt.figure(figsize=(max(8, T*0.5), 4))
plt.imshow(routing.numpy(), aspect='auto', cmap='tab10', vmin=0, vmax=moe_model.blocks[0].ffn.n_experts - 1)
plt.xticks(range(T), tokens, rotation=45, ha='right')
plt.yticks(range(L), [f"layer {i}" for i in range(L)])
plt.colorbar(label="chosen expert id", ticks=range(moe_model.blocks[0].ffn.n_experts))
plt.title("Expert routing per token x layer \u2014 watch for vertical stripes (specialization)")
plt.tight_layout(); plt.show()'''
    ),
    md(
        """## 5. Active vs total parameter comparison"""
    ),
    code(
        '''def expert_param_count(d_model, d_ff):
    return 3 * d_model * d_ff  # gate, up, down


total = sum(p.numel() for p in moe_model.parameters())
per_expert = expert_param_count(cfg.d_model, cfg.d_ff)
ffn_total_per_block = moe_model.blocks[0].ffn.n_experts * per_expert
ffn_total = ffn_total_per_block * cfg.n_layers

# A "dense equivalent" with the same total params would use ffn_total / 1 expert per block.
print(f"MoE total params:        {total:,}")
print(f"  of which FFN:           {ffn_total:,}  (across {cfg.n_layers} layers x {moe_model.blocks[0].ffn.n_experts} experts)")
print(f"Active params per token: ~ total - (n_experts-1) * params_per_expert * n_layers")
active = total - (moe_model.blocks[0].ffn.n_experts - 1) * per_expert * cfg.n_layers
print(f"  -> active per token:    {active:,}  ({100*active/total:.1f}% of total)")
print(f"This is why MoE is described as bigger params at same FLOPs.")'''
    ),
    md(
        """## 6. What you just did

- Built an MoE block (gate + experts + load-balance loss).
- Built and briefly trained an MoE LLM on TinyStories.
- Visualized expert specialization on a real sentence.
- Computed the active-vs-total parameter ratio that explains why MoE is the frontier-of-frontier architecture.

**Next: Post 11 \u2014 SFT and DPO.** We turn our base model into an instruction-following assistant the way Claude and Gemini are turned into chatbots."""
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
