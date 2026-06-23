# Building GPT From Scratch

## Chapter 11

# Inference Optimization

> *"Training is done once. Inference is done a billion times. Every microsecond you save in the forward pass is a billion microseconds saved."*

---

## 11.1 The Cost of Inference

Training is compute-bound: the bottleneck is how fast the GPU can multiply matrices. Inference at scale is *latency-bound*: the bottleneck is how fast you can produce the first token, and then each subsequent token.

The first token from a transformer requires the full forward pass over the entire prompt. Each subsequent token is one step. So for a 100-token response:

- Token 1: forward pass over the full prompt
- Token 2: forward pass over prompt + 1
- Token 100: forward pass over prompt + 99

Naively, that's `100 * (prompt + 100)²` of attention work. For a 2000-token prompt and 100-token response, that's 4.2 billion attention entries. Most of them are *recomputed* for every new token.

**KV caching** eliminates the recomputation by saving the key and value vectors from previous tokens. After generating the first token, every subsequent token only needs to compute Q for the new token, K and V for the new token, and attend to the cached K/V from all previous tokens. The attention cost becomes linear in sequence length instead of quadratic.

This single optimization is the difference between "the model can generate text" and "the model can serve millions of users at a reasonable cost." It's mandatory in production.

---

## 11.2 KV Caching

Recall: attention is `softmax(QKᵀ/√d) V`. For a new token, we compute its Q, K, V vectors. The Q for the new token attends to all the Ks (the new one + the cached ones) and the Vs (new one + cached). The result is a vector for the new position only.

```python
def forward_with_cache(self, x, cache=None):
    """x: [B, T_new] — just the new tokens, not the full sequence.
    cache: dict with 'k' and 'v' of shape [B, n_head, T_past, d_head]
           (None on first call, then updated and returned).
    """
    B, T_new = x.shape

    # Project the new tokens
    Q_new = self.W_q(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
    K_new = self.W_k(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
    V_new = self.W_v(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)

    # Concatenate with the cache (None on first call → just use K_new, V_new)
    if cache is not None:
        K = torch.cat([cache["k"], K_new], dim=2)
        V = torch.cat([cache["v"], V_new], dim=2)
    else:
        K, V = K_new, V_new

    # Save the *new* K, V into the cache for next time
    new_cache = {"k": K.detach(), "v": V.detach()}

    # Attention: Q_new @ K^T / sqrt(d)
    scores = Q_new @ K.transpose(-2, -1) / (self.d_head ** 0.5)
    # Causal mask only for the new positions
    T_past = K.size(2) - T_new
    mask = torch.triu(
        torch.ones(T_new, K.size(2), device=x.device), diagonal=T_past + 1
    ).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = (weights @ V).transpose(1, 2).contiguous().view(B, T_new, C)
    return self.W_o(out), new_cache
```

Key things to notice:

1. **The Q projection is only computed for the new tokens.** T_new is usually 1 (single token generation) or a few (parallel sampling).
2. **The K and V projections are also only for new tokens**, but they're *appended* to the cache.
3. **The attention scores are `[B, n_head, T_new, T_total]`** where T_total = T_past + T_new. On autoregressive generation, T_new = 1, so this is a 1×T_total attention — much cheaper than the uncached T_total×T_total.
4. **The cache is detached** so gradients don't flow into it. We never backprop through the cache during generation.

For a 2000-token prompt with 100-token response:
- Uncached: `~ 4.2 billion` attention entries
- Cached: `~ 2.1 million` attention entries — **2000× less work**

---

## 11.3 Causal Mask with Cache

The trickiest part of KV caching is the causal mask. When generating, the new token can attend to all past positions (no future tokens exist). But if you're computing multiple new tokens at once (e.g., during a teacher-forced forward pass), the mask needs to be more careful.

The convention: `mask[i, j] = True` means position `i` (in the new tokens) is blocked from attending to position `j` (in the total sequence including cached).

For T_new = 1 (autoregressive generation), the mask is `[1, T_total]` where:
- Row 0 attends to all columns 0..T_past (cached tokens)
- It is blocked from column T_past (itself in the new tokens, but that's fine — we only block future)

Actually for a single new token, the mask is empty (the new token attends to all cached + itself). No mask needed.

For T_new > 1 (parallel), the mask is:
- Row `i` in [0, T_new) is blocked from columns `[T_past + i + 1, T_past + T_new)` (the future new tokens)

```python
def causal_mask(T_new, T_past, device):
    """Returns a [T_new, T_past + T_new] mask, True = blocked."""
    # Position (i_new, j_total) is blocked if j_total > T_past + i_new
    i = torch.arange(T_new, device=device).unsqueeze(1)   # [T_new, 1]
    j = torch.arange(T_past + T_new, device=device).unsqueeze(0)  # [1, T_total]
    return j > (T_past + i)
```

---

## 11.4 Batched Generation

For serving multiple users at once, you batch independent generation requests. Each request has its own cache, and the caches are padded to the longest length in the batch.

```python
class KVCache:
    """One cache per request. Stored as a dict of tensors per layer."""

    def __init__(self, batch_size, n_layer, n_head, d_head, max_seq_len, device):
        # Per-layer cache, one tensor of shape [B, n_head, 0, d_head]
        # (will be grown as tokens are added)
        self.k = [torch.empty(batch_size, n_head, 0, d_head, device=device)
                  for _ in range(n_layer)]
        self.v = [torch.empty(batch_size, n_head, 0, d_head, device=device)
                  for _ in range(n_layer)]

    def update(self, layer_idx, k_new, v_new):
        """Append new K, V for a layer. Returns the full K, V for that layer."""
        self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
        self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]
```

For variable-length requests, the cache has "wasted" slots for shorter sequences. Production systems use **paged attention** (vLLM) which eliminates this waste by storing cache pages in a virtual memory layout.

---

## 11.5 Continuous Batching

Naive serving: wait for all requests in a batch to finish before starting the next batch. This means some requests are sitting idle while the longest one finishes.

**Continuous batching** (used by vLLM, TGI, TensorRT-LLM): when one request in a batch finishes, replace it with a new request. The batch is constantly full of active requests.

This typically gives **2-3× throughput** improvement over static batching, with no latency impact on the requests themselves.

---

## 11.6 Quantization

Most of the model parameters are in float32 (4 bytes) or float16 (2 bytes). The activations and weights have a lot of redundancy — many weights have values clustered in a small range. We can represent them with fewer bits.

**4-bit quantization** (the standard for production) shrinks the model to 1/4 of its fp16 size with minimal quality loss. The math:

- **fp16**: 5-bit exponent + 10-bit mantissa = lots of precision
- **int8**: 8-bit uniform range = ~0.5% precision
- **int4**: 4-bit uniform range = ~6% precision
- **NF4 (4-bit NormalFloat)**: 4-bit with non-uniform spacing tuned for normally-distributed weights. ~3% precision, much better than uniform int4.

```python
# Toy example: quantize a weight matrix to 4-bit NF4
import torch

def quantize_nf4(weight):
    """Toy NF4 quantization — real implementation has more careful handling."""
    # NF4 levels (the standard set)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
         0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    # Symmetric quantize: scale to [-1, 1], find nearest NF4 level
    absmax = weight.abs().max()
    normalized = weight / absmax
    # Find the nearest NF4 level (real impl uses a lookup)
    indices = torch.searchsorted(nf4_levels, normalized.flatten())
    indices = indices.clamp(0, 15)
    quantized = nf4_levels[indices].view_as(weight)
    return quantized * absmax   # dequantized
```

The savings are enormous:
- 7B model in fp16: 14 GB
- 7B model in 4-bit: 3.5 GB

This is why a 70B model can run on a single 24GB consumer GPU.

**Quality cost.** With proper calibration, 4-bit NF4 quantization loses <0.5% accuracy on most benchmarks. 8-bit loses essentially nothing.

---

## 11.7 Speculative Decoding

LLMs generate one token at a time because each token depends on the previous. But what if a smaller, faster model could predict the next few tokens, and the large model could verify them in parallel?

**Speculative decoding**: a small "draft" model (e.g., 1B params) generates K candidate tokens. The large "target" model (e.g., 70B) evaluates all K in a single forward pass. If they match, all K are accepted. If not, the position of the first mismatch is the new token, and we repeat.

The result: **2-3× faster generation** with no quality change, because the output distribution is identical to what the target model would have produced.

The key insight: evaluating 5 candidate tokens in parallel costs the same as evaluating 1 token (one forward pass through the target model). If the small model is right 60% of the time per token, the expected number of accepted tokens per round is `1 + 0.6 + 0.36 + 0.22 + 0.13 = 2.31`. So we generate ~2.3 tokens per round instead of 1 — a 2.3× speedup.

This works in production for any pair of models where one is a strict subset of the other. The Llama 3.1 series (8B, 70B, 405B) is designed for speculative decoding: the 8B model is the draft, the 70B is the target, the 405B is the oracle.

---

## 11.8 Flash Attention

The standard attention implementation materializes the full `[T, T]` attention matrix in memory, computes softmax over it, then multiplies by V. This is O(T²) memory and O(T²) compute.

**Flash Attention** (Tri Dao et al., 2022) reorders the computation to never materialize the full matrix. It processes the sequence in blocks, computes partial softmaxes per block, and combines them. The result:

- **Same FLOP count** (the math is identical)
- **O(T) memory** (only the current block lives in SRAM/cache)
- **2-4× faster** in practice, just from better memory access patterns

Flash Attention 2 (2023) further optimizes the block sizes and pipelining. Flash Attention 3 (2024) adds fp16-specific optimizations for Hopper GPUs.

This is why a 200K-token context model is now feasible — the attention memory scales linearly instead of quadratically.

---

## 11.9 Paged Attention (vLLM)

Even with KV caching, a serving system has to allocate a contiguous memory block for each request's cache. For variable-length requests, this wastes memory on padding.

**Paged Attention** (vLLM, Kwon et al., 2023) treats the KV cache like virtual memory: each request's cache is split into fixed-size "pages" (typically 16 tokens per page), and the pages are stored in a non-contiguous pool. A page table maps each request's logical positions to physical pages.

This eliminates fragmentation, allows near-100% memory utilization, and enables features like prefix sharing across requests. The result: **2-4× throughput** improvement over naive KV cache allocation.

---

## 11.10 The Inference Stack

A production inference stack has many layers. Here's what a typical request goes through:

```
Client request
  → Load balancer (round-robin or least-loaded)
    → API server (FastAPI, TorchServe)
      → Tokenizer (text → tokens)
        → Continuous batch scheduler (vLLM, TGI)
          → Per-request KV cache (paged)
            → Model with Flash Attention + quantization
              → Speculative decoding (optional)
                → Detokenizer (tokens → text)
                  → Response
```

Each layer can be optimized independently. The most impactful are: continuous batching, Flash Attention, and quantization. Together they give 10-100× throughput improvement over a naive implementation.

---

## 11.11 Latency Budgets

A typical user request has these latency targets:

| Stage | Target | Why |
|---|---|---|
| Network round-trip | 50ms | Internet round-trip |
| Queue + scheduling | 5ms | Should be near-instant |
| Pre-fill (prompt forward) | 50-200ms | O(prompt length) |
| First token | pre-fill + queue | User-visible |
| Per-token generation | 10-30ms | Streaming response |
| Total response | as long as it takes | Streaming makes this feel fast |

**Time to first token (TTFT)** is what the user waits for after sending the prompt. This is dominated by the pre-fill: a single forward pass over the entire prompt.

**Inter-token latency (ITL)** is the gap between consecutive tokens during streaming. This is what makes the response feel smooth — 30ms per token = 33 tokens/sec.

**Tokens per second (TPS)** is the throughput measure. For a 70B model on an A100, you get 30-50 TPS per request; with batching, the aggregate can be 1000+ TPS per GPU.

---

## 11.12 Batching and Throughput

The relationship between batch size and throughput is roughly linear up to a point, then plateaus:

```
batch=1:   40 tokens/sec/request
batch=8:   38 tokens/sec/request   (304 total)
batch=32:  35 tokens/sec/request   (1120 total)
batch=128: 20 tokens/sec/request   (2560 total)
batch=512:  5 tokens/sec/request   (2560 total)  ← plateaus
```

The GPU has a fixed amount of compute per second. At small batch sizes, the GPU is underutilized. At large batch sizes, the per-request latency is dominated by the GPU's compute, so per-request TPS drops — but the aggregate stays roughly constant.

The optimal batch size depends on:
- **GPU memory** (KV cache size = batch × seq_len × 2 × n_layer × d_model)
- **Latency SLA** (larger batches = higher per-request latency)
- **Traffic pattern** (you need enough requests to fill the batch)

A typical production setup runs at batch 32-128, with hundreds of tokens/sec/request and 1000+ aggregate TPS.

---

## 11.13 Sampling Parameters and Latency

Sampling parameters don't change the compute cost (per token), but they change the *quality* and *perceived speed*:

- **`temperature=0` (greedy)**: deterministic, fastest. Always picks the argmax. Best for code generation.
- **`temperature=0.7-1.0` (typical)**: adds randomness. Default for most chat models.
- **`temperature=2.0+` (chaos)**: nearly uniform sampling. Useful for creative writing.

`top_k` and `top_p` (nucleus) filter the distribution but don't change compute. The model's logits are computed for all 50K vocab tokens regardless; you just take a max or threshold after.

**Speeding up sampling**: the softmax + multinomial step is fast (microseconds). The bottleneck is the model forward pass, not the sampling. Don't try to optimize sampling — it doesn't matter.

---

## 11.14 What You Should Care About

If you're deploying a model in production, the optimizations in priority order are:

1. **Use a good serving system** (vLLM, TGI, TensorRT-LLM) — they bundle continuous batching, paged attention, Flash Attention, and quantization.
2. **Quantize to 4-bit or 8-bit** — minimal quality loss, big memory savings.
3. **Use KV caching** — mandatory, not optional.
4. **Match batch size to traffic** — too small wastes GPU, too large hurts latency.
5. **Stream responses** — TTFT matters less than ITL for user experience.
6. **Consider speculative decoding** — 2-3× speedup with no quality cost, but adds complexity.

Things you can ignore unless you have a specific reason:
- Custom CUDA kernels (the serving systems already do this)
- Custom sampling (the bottleneck is the forward pass)
- Model architecture changes (use a different model)

---

## Chapter Summary

- **KV caching** reduces attention cost from O(T²) per token to O(T). Mandatory for production.
- **Causal mask with cache** is the trickiest part — for autoregressive generation, no mask is needed; for parallel generation, mask only the future new positions.
- **Continuous batching** gives 2-3× throughput improvement over static batching.
- **Quantization to 4-bit** shrinks models to 1/4 their fp16 size with <0.5% quality loss.
- **Speculative decoding** uses a small model to draft, a large model to verify in parallel. 2-3× speedup with no quality change.
- **Flash Attention** processes attention in blocks to avoid materializing the full matrix. 2-4× speedup from better memory access.
- **Paged attention** (vLLM) treats KV cache like virtual memory. 2-4× throughput improvement.

The inference stack is where ML engineering meets systems engineering. A naive PyTorch forward pass is maybe 1% of what a production system achieves.

In Chapter 12, we cover **deployment** — packaging the model, exposing it as an API, monitoring, and the operational concerns of running it in production.

---

## Exercises

1. **KV cache speedup.** Implement a cached and uncached attention layer. Measure the speedup on a 2000-token sequence with 100 generated tokens. Expected: 10-50× for long contexts.
2. **Batch size sweep.** Run a fixed inference workload at batch sizes {1, 4, 16, 64, 256}. Plot per-request TPS and aggregate TPS. Where does aggregate TPS plateau?
3. **Quantization quality.** Load a model in fp16 and 4-bit, run a fixed eval, compare perplexity. Expected: <0.1 perplexity increase for 4-bit NF4.
4. **Speculative decoding.** Use a 1B model as draft and a 7B model as target. Measure the average acceptance rate. Expected: 50-70% for similar models.
5. **Time to first token.** Measure TTFT for prompts of length {10, 100, 1000, 10000}. Confirm it scales linearly with prompt length.
6. **Continuous batching.** Implement a simple continuous batch scheduler. Compare to a static batch scheduler on the same request mix.

The full implementation lives in `code/chapter11/inference.py` — a KV-cached attention layer, a generation loop with cache, and a benchmark comparing cached vs uncached.