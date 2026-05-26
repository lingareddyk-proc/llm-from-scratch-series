# Post 9 — Sampling, KV-Cache, and Making Inference Fast

*Part 9 of 12 in **LLM From Scratch**, a series that builds a Gemini/Claude-style decoder in Google Colab, one component at a time.*

We trained a model in Post 8. Generating from it is *slow* — every new token re-runs the full forward pass over the entire prefix. For a 200-token completion, that means computing attention over positions `[0..0]`, then `[0..1]`, then `[0..2]`, … up to `[0..200]`: an O(T²) waste.

This post fixes that with the **KV-cache**, the single most important inference optimization in modern LLM serving. We also cover the **sampling strategies** every chatbot uses — temperature, top-k, top-p, min-p — and show how they shape the output style.

## How to read this post

- **Skim (3 min):** the speedup chart in §7.
- **Read (12 min):** all of it.
- **Run (15 min):** open the Colab. The before/after `tokens/sec` benchmark is the highlight.

> Colab: **[Open in Colab](https://colab.research.google.com/github/lingareddyk-proc/llm-from-scratch-series/blob/main/posts/09-sampling-kvcache/notebook.ipynb)** — free CPU or T4, ~10 min.

---

## 1. Why naive generation is slow

To generate token `t`, we need the attention output at position `t-1`. But our `model(ids)` runs the whole forward pass over positions `0..t-1`, recomputing:

- All Q vectors (necessary)
- All K vectors (already computed in the previous step)
- All V vectors (already computed)
- All attention scores (already computed)

That's a lot of redundant work. For a sequence of length `T`, we do `1 + 2 + 3 + ... + T = O(T²)` work, when we should be doing `O(T)`.

## 2. The KV-cache idea

Cache `K` and `V` for every layer × every head × every position we've seen. When generating the next token:

1. Embed only the **new** token.
2. Compute the **new Q, K, V** for that one position.
3. **Append** the new K and V to the cache (so the cache grows by one).
4. Attention: `softmax(Q_new · K_cache.T / √d) · V_cache` — but now Q is `(B, H, 1, d_head)` and K/V are `(B, H, T, d_head)`. The attention computation is `O(T)`, not `O(T²)`.

Per generation step, we do **constant work for the new token + linear work to attend to history**, instead of redoing everything.

## 3. Why this is "KV" and not "QKV" cache

Q is a *fresh* projection of the new token's residual stream. We compute it anew every step (and discard it). K and V live in the cache because *they describe the past* — they don't change once written.

This asymmetry is also why **Grouped-Query Attention (GQA)** matters so much: GQA lets us share K and V across multiple Q-heads, shrinking the cache by 4–8× without changing the compute much. The KV-cache size is the **biggest memory bottleneck for serving** at frontier scale.

For Llama 3 70B at 128k context with batch size 1: the KV-cache alone is ~40 GB. With MHA (no GQA) it would be ~320 GB.

## 4. The diagram

```mermaid
flowchart LR
    subgraph naive [Naive generation: O(T^2)]
        N1["embed [0..t-1]"] --> N2["compute Q,K,V for all positions"] --> N3["full attention"] --> N4[next token]
        N4 -.append.-> N1
    end
    subgraph cached [KV-cache: O(T)]
        C1["embed [t-1] only"] --> C2["compute Q,K,V for ONE position"]
        C2 --> C3["append K,V to cache"]
        C3 --> C4["attention with cached K,V"]
        C4 --> C5[next token]
        C5 -.append.-> C1
    end
```

## 5. Sampling strategies that matter

We've used temperature + top-k since Post 1. Here are the others worth knowing.

| Strategy   | What it does                                                                                  | When to use                                         |
| ---------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Greedy     | Always pick top-1. `T=0`.                                                                     | Deterministic / tasks needing one right answer.     |
| Temperature| Divide logits by `T` before softmax.                                                          | Universal first knob.                               |
| Top-k      | Keep only top `k` candidates, renormalize.                                                    | Prevent very low-prob "weird" tokens.               |
| Top-p (nucleus) | Keep smallest set whose cumulative prob ≥ `p`. Cardinality adapts to distribution sharpness. | Modern default in most chat models.                 |
| Min-p      | Keep tokens with prob ≥ `min_p × prob(top1)`. Adaptive like top-p, simpler to reason about.   | New (2024+); often outperforms top-p at low temps.  |
| Repetition penalty | Down-weight already-seen tokens.                                                       | Stop "I am very very very very" loops.              |

Frontier models (Gemini, Claude, GPT-4o) typically use `T=0.7-1.0` + `top_p=0.95` or `min_p=0.05` as defaults.

## 6. Speculative decoding (preview)

Beyond what we'll implement: at frontier scale, every API uses **speculative decoding** — a small "draft" model proposes 5–10 tokens at once, and the big model verifies them in a single batched forward pass. Net: 2–4× more tokens/sec at no quality loss. Same KV-cache, more tokens per cache traversal. Out of scope for this post but worth knowing the name.

## 7. What you'll do in the Colab

1. Add a KV-cache to our `GroupedQueryAttention` (a small modification).
2. Build a `generate_with_cache` function and a `generate_naive` baseline.
3. Load the trained checkpoint from Post 8 (or use a freshly-initialized model on CPU if you don't have the checkpoint).
4. **Benchmark** tokens/sec on a 200-token completion — naive vs cached.
5. Try **top-p** and **min-p** samplers on the same prompt and compare outputs.
6. Sweep temperature and visualize how output entropy changes.

Expected speedup: 5–15× on CPU, 10–30× on GPU for a 200-token completion. Larger for longer prompts.

## 8. How frontier LLMs do this

- **KV-cache:** every production LLM uses it. Always.
- **PagedAttention** (vLLM): manages the cache in fixed-size "pages" so many users share GPU memory. This is how Anthropic and OpenAI serve millions of requests on the same accelerators.
- **Speculative decoding:** used by every major API (OpenAI confirmed; Anthropic confirmed; Google strongly implied).
- **Quantized cache** (e.g. KV-cache in fp8 or int4): another 2–4× memory reduction at small quality cost.
- **Continuous batching:** the scheduler streams in new requests as old ones finish, keeping the GPU at high utilization.

The line between "an LLM" and "an LLM you can serve cheaply" is mostly KV-cache engineering on top of the same model.

## 9. What we'll do next post

We've covered training and serving for a "dense" model — every parameter activates for every token. Frontier labs increasingly use **Mixture-of-Experts** to get bigger models at the same FLOPs. Gemini, GPT-4, and the Mixtral family are MoE. Post 10 swaps our SwiGLU FFN for an MoE block and watches experts specialize.

**Next: Post 10 — Mixture of Experts.**

---

*Code: [github.com/lingareddyk-proc/llm-from-scratch-series](https://github.com/lingareddyk-proc/llm-from-scratch-series), tagged `v0.9.0`.*
