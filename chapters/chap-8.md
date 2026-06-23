# Building GPT From Scratch

## Chapter 8

# GPT Architecture

> *"We've built every piece. Now we put them together, parameterize the whole thing with a config dict, and write a single function that turns 'text in' into 'text out'."*

---

## 8.1 From Blocks to GPT

A GPT is just the `Transformer` model from Chapter 7, plus:

1. A **config** that names all the hyperparameters
2. A **forward pass** that returns the loss directly (not just the logits)
3. A **generation loop** that turns logits into new tokens

That's it. The "GPT architecture" is the name we give this specific combination. It's not a new idea — it's the synthesis of everything in Chapters 4–7 with the right bookkeeping.

This chapter builds:

- A `GPTConfig` dataclass
- A `GPT` model that takes a config and produces logits
- A `forward` that returns `(logits, loss)`
- A `generate` method that autoregressively samples
- A `from_pretrained` helper for loading OpenAI's GPT-2 weights
- A small training script that ties the whole pipeline together

By the end you'll be able to type:

```python
config = GPTConfig(block_size=256, vocab_size=50257, n_layer=6, n_head=6, n_embd=384)
model = GPT(config)
out = model.generate("Once upon a time", max_new_tokens=50)
```

…and get a coherent (if mostly nonsense) completion from a model you built from scratch.

---

## 8.2 The GPTConfig

Everything about the model — depth, width, vocabulary, context length — is a number. We bundle them into a config:

```python
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024    # max context length (T)
    vocab_size: int = 50257   # GPT-2's BPE vocab size
    n_layer: int = 12         # number of transformer blocks
    n_head: int = 12          # number of attention heads
    n_embd: int = 768         # embedding dimension (d_model)
    dropout: float = 0.0      # dropout rate
    bias: bool = True         # use bias in Linears and LayerNorms?
```

Two design decisions worth flagging:

**`block_size`** is the maximum context length. Sequences longer than this will be truncated. Larger = more memory (attention is O(T²)).

**`bias: bool = True`** follows the GPT-2 convention. Modern models (Llama) often set this to `False` for a small parameter savings and slight speedup. We'll keep it `True` for compatibility with GPT-2 weights.

---

## 8.3 The GPT Model

Here's the entire architecture. It's a few dozen lines because all the heavy lifting is in the `TransformerBlock` from Chapter 7.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Component 1: token + position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                TransformerBlock(config.n_embd, config.n_head,
                                config.block_size, config.dropout)
                for _ in range(config.n_layer)
            ]),
            ln_f = RMSNorm(config.n_embd),   # final norm before the head
        ))

        # Component 2: output head (tied to token embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input and output share the same matrix
        self.transformer.wte.weight = self.lm_head.weight

        # Init: small std for all linear layers, RMSNorm gamma=1
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.gamma)

    def forward(self, idx, targets=None):
        # idx: [B, T] token IDs
        # targets: [B, T] token IDs (or None for inference)
        B, T = idx.shape
        assert T <= self.config.block_size

        # 1. Embeddings
        pos = torch.arange(T, device=idx.device)               # [T]
        tok_emb = self.transformer.wte(idx)                      # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)                      # [T, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)             # [B, T, n_embd]

        # 2. Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 3. Final norm
        x = self.transformer.ln_f(x)

        # 4. Output projection (with weight tying)
        logits = self.lm_head(x)                                 # [B, T, vocab_size]

        # 5. Loss (if training)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: [B, T] context tokens
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Take the logits for the last position only
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

That's a complete GPT. The model is 8 levels deep: config → components → forward → loss/generate. Every modern decoder-only transformer fits this skeleton.

---

## 8.4 The Forward Pass, Step by Step

Walk through what happens when you call `model(idx)` with `idx` of shape `[B, T]`:

```
Input: idx    shape [B, T] = [4, 1024]      (4 sequences, 1024 tokens each)
                │
                ├──► wte(idx)              [B, T, n_embd]   (token lookup)
                └──► wpe(arange(T))        [T, n_embd]      (position lookup)
                │
                ▼
            tok_emb + pos_emb             [B, T, n_embd]   (broadcast add)
                │
                ▼
            dropout
                │
                ▼
            ┌──────────────────────┐
            │  TransformerBlock × N │       [B, T, n_embd]   (same shape, refined)
            └──────────────────────┘
                │
                ▼
            RMSNorm
                │
                ▼
            lm_head                       [B, T, vocab_size]
                │
                ▼
            cross_entropy(targets)        scalar loss
```

Notice: **the shape never changes** inside the blocks. Each block is a residual function with the same input and output shape. All the work is done in the dimensions you can't see (the `[B, T, n_embd]` tensor's values are getting more refined at every block).

---

## 8.5 The Generation Loop

`generate` is the inference loop. It's autoregressive: each new token is produced one at a time, conditioned on everything before it.

```
context:    "The cat sat"
                     ▼
              model → softmax → sample → "on"
context:    "The cat sat on"
                     ▼
              model → softmax → sample → "the"
context:    "The cat sat on the"
                     ▼
              model → softmax → sample → "."
context:    "The cat sat on the."
                     ▼
            ...continue until max_new_tokens
```

Three tricks make this work in practice:

1. **KV caching** (Chapter 11): don't recompute K, V for tokens already in the context. Each new token only needs to attend to the cached past.
2. **Temperature**: `logits / T` before softmax. `T < 1` makes the distribution sharper (more confident); `T > 1` makes it flatter (more random).
3. **Top-k sampling**: only consider the top `k` most likely tokens, set the rest to `-inf`. Prevents the model from sampling nonsensical low-probability tokens.

```python
# High temperature = creative, low = conservative
out_greedy = model.generate(idx, max_new_tokens=20, temperature=0.0)   # always pick argmax
out_random = model.generate(idx, max_new_tokens=20, temperature=1.0)  # sample from full distribution
out_focused = model.generate(idx, max_new_tokens=20, temperature=0.7, top_k=40)
```

`temperature=0.0` is the convention for greedy decoding (argmax), even though mathematically it would be undefined. Our `generate` handles it via the `argmax`-equivalent of always picking the top probability.

---

## 8.6 Parameter Count for the Family

Using the `GPTConfig`, the model size is fully determined by `(n_layer, n_head, n_embd, vocab_size)`. Common configurations:

| Model | n_layer | n_head | n_embd | vocab | params |
|---|---|---|---|---|---|
| nanoGPT char-level | 6 | 6 | 384 | 65 | ~10M |
| nanoGPT BPE (our SLM) | 6 | 6 | 384 | 50257 | ~30M |
| GPT-2 small | 12 | 12 | 768 | 50257 | 124M |
| GPT-2 medium | 24 | 16 | 1024 | 50257 | 354M |
| GPT-2 large | 36 | 20 | 1280 | 50257 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 50257 | 1558M |

For our Small Language Model in this book, we'll use the **nanoGPT** BPE config: 6 layers, 6 heads, 384 dims, 1024 context, 50257 vocab. About 30M parameters (most of them in the embedding table — the model itself is only ~10M). Trains in a few hours on a single GPU, in a few minutes on a laptop if you have a recent Apple Silicon chip.

```python
config = GPTConfig(
    block_size=1024,
    vocab_size=50257,    # GPT-2's BPE tokenizer
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=False,
)
model = GPT(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Parameters: 29.94M (or so — depends on weight tying and bias)
```

---

## 8.7 Loading Pretrained Weights

Once you have the architecture right, loading OpenAI's GPT-2 weights is a matter of mapping the right tensors. The state dict keys follow a strict pattern:

```
transformer.wte.weight           # token embedding
transformer.wpe.weight           # position embedding
transformer.h.0.ln_1.weight      # block 0, pre-attn RMSNorm gamma
transformer.h.0.attn.W_q.weight  # block 0, query projection
transformer.h.0.attn.W_k.weight  # block 0, key projection
transformer.h.0.attn.W_v.weight  # block 0, value projection
transformer.h.0.attn.W_o.weight  # block 0, output projection
transformer.h.0.ln_2.weight      # block 0, pre-FFN RMSNorm gamma
transformer.h.0.mlp.W_gate.weight  # (Llama) or c_fc.weight (GPT-2)
transformer.h.0.mlp.W_up.weight    # (Llama) or c_fc.weight (GPT-2) — only one for GPT-2
transformer.h.0.mlp.W_down.weight  # (Llama) or c_proj.weight (GPT-2)
transformer.ln_f.weight          # final RMSNorm
lm_head.weight                   # output head (tied to wte)
```

If we want to load GPT-2's weights, the mapping is direct. If we want to load Llama, we need to handle the FFN name difference (`c_fc` → split into `W_gate` and `W_up`).

```python
def from_pretrained(model, hf_model_name="gpt2"):
    """Load OpenAI's GPT-2 weights into our model."""
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_state = hf.state_dict()

    # Map HF keys to our keys
    our_state = model.state_dict()
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                  'mlp.c_fc.weight', 'mlp.c_proj.weight']

    for k in our_state:
        if k.startswith('transformer.h.') and any(k.endswith('.' + t) for t in transposed):
            # HF stores conv1d weights as [in, out], we want [out, in]
            our_state[k].copy_(hf_state[k].t())
        elif k in hf_state:
            our_state[k].copy_(hf_state[k])
        else:
            # weight tying means lm_head == wte
            pass

    model.load_state_dict(our_state)
    return model
```

This is the code `nanoGPT` ships. With a config that matches GPT-2 small (`n_layer=12, n_head=12, n_embd=768, vocab_size=50257`), you can `from_pretrained(model, "gpt2")` and immediately generate text with OpenAI's actual weights. The architecture really is that close.

---

## 8.8 The Full Pipeline: Text In, Text Out

End-to-end, the GPT inference pipeline is:

```
"The cat sat"
        │
        ▼
   GPT tokenizer (Chapter 3)
        │  encode: text → [15496, 11, 3797, 7731, 13]  (or however your BPE splits it)
        ▼
   GPT model.forward(idx)
        │  [B, T, vocab_size] logits
        ▼
   softmax(logits / T)
        │  probabilities
        ▼
   sample
        │  one new token ID
        ▼
   append to context, repeat
        │
        ▼
   GPT tokenizer.decode
        │
        ▼
   "The cat sat on the mat."
```

The tokenizer is the only piece outside the model. It's the bridge between "the user's world" (text) and "the model's world" (integer IDs). When you build this pipeline correctly, the model can read and write text just like any LLM you've ever used.

---

## 8.9 Common Bugs and How to Avoid Them

After building a few of these, you'll find that 90% of the time your GPT doesn't train, it's one of these:

| Bug | Symptom | Fix |
|---|---|---|
| Off-by-one in position embeddings | Loss spikes after `block_size` tokens | Pass `arange(T)` not `arange(0, T+1)`; check `block_size ≥ T` |
| Forgot to tie `lm_head` to `wte` | Model has 2× the embedding params; loss is fine but training is slower | `lm_head.weight = wte.weight` |
| Used `F.cross_entropy` without `ignore_index=-100` | Loss is averaged over padding tokens, looks too low | Add `ignore_index=-100` |
| Causal mask inverted | Loss explodes, model produces garbage | Verify `triu(ones(T, T), diagonal=1)` is the *upper* triangle |
| Position embedding learned but `block_size` too small | Sequence gets cut off | Set `block_size` to your dataset's max length + a few hundred tokens |
| Learning rate too high | Loss spikes immediately, recovers slowly or never | Use 3e-4 to 1e-3 for nanoGPT-scale models; 1e-4 to 3e-4 for 100M+ models |
| Forgot to set model to `.train()` mode | Dropout is off in training mode (or on in eval mode) | Call `model.train()` before training, `model.eval()` before generating |

The most common single bug is **forgetting the causal mask**. Without it the model can "cheat" by looking at the answer during training. Loss will look fine, generations will be a copy of the prompt.

---

## 8.10 What the Forward Pass Really Computes

Mathematically, the forward pass is a factorization of a probability distribution over text. The model computes:

```
P(token_t | token_1, ..., token_{t-1})  for every t in 1..T
```

The loss is the **negative log-likelihood** of the actual next token, averaged over all positions:

```
L = -1/T * sum_t  log P(actual_token_t | tokens_1..t-1)
```

This is also called the **cross-entropy** between the model's predicted distribution and the true next token. Minimizing it is exactly equivalent to maximizing the likelihood of the training corpus under the model.

If your training data is `D = [d_1, d_2, ..., d_N]` and your model is `p_θ`, training minimizes:

```
L(θ) = -1/N * sum_i  log p_θ(d_i | d_<i)
```

That's the entire training objective. The architecture exists to parameterize `p_θ` in a way that's expressive, scalable, and trainable.

---

## Chapter Summary

- A **GPT** is a stack of transformer blocks, plus token + position embeddings and a tied output head.
- The `forward` pass returns `(logits, loss)`. Pass `targets=None` for inference, real targets for training.
- The `generate` method is an autoregressive sampling loop with optional temperature and top-k.
- All the hyperparameters live in a `GPTConfig` dataclass.
- Loading pretrained weights is a matter of mapping state-dict keys.
- The model is `P(token_t | tokens_<t)` parameterized by attention and feed-forward layers.

In Chapter 9, we train one. From scratch, on real text, watching the loss go down.

---

## Exercises

1. **Config sweep.** Build models with `n_layer ∈ {2, 4, 6, 12}`, `n_embd ∈ {128, 256, 384, 768}`. For each, print parameter count and a single forward pass's FLOPs (rough: `6 * n_params * T` per token).
2. **Generate from random.** Build a fresh model and call `generate`. The output will be garbage — that's expected. Try several seeds. Is the output reproducible?
3. **Generation parameters.** Set `temperature ∈ {0.0, 0.5, 1.0, 2.0}` and `top_k ∈ {None, 1, 10, 100}`. How do the outputs differ in style (conservative vs. creative)?
4. **KV cache stub.** Modify `generate` to keep a running K, V cache. Verify that the output is identical to the no-cache version. (We'll fill this in properly in Chapter 11.)
5. **Load GPT-2.** Install `transformers` (`pip install transformers`). Load `gpt2` (small) into a matching `GPT` model. Generate text from the prompt `"In a hole in the ground there lived a"`. Does the output look like Tolkien?
6. **Count FLOPs.** For a forward pass on `T = 1024` tokens with the nanoGPT config, estimate the total FLOPs. (Hint: each matmul is `2 * m * n * k` FLOPs.)
7. **Speed test.** Time 100 forward passes at `B=4, T=1024` on your machine. How many tokens per second can your model process?

The full implementation lives in `code/chapter08/gpt.py` — a complete, training-ready GPT. Run it to see the parameter breakdown, try `from_pretrained` if you have `transformers` installed, and start thinking about what data you'd want to train it on.