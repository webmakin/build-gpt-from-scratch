"""
Chapter 8: GPT Architecture

Covers:
  8.2  GPTConfig dataclass
  8.3  The full GPT model (config + forward + generate)
  8.4  Forward pass walkthrough
  8.5  Generation loop with temperature and top-k
  8.6  Parameter count for the family (nanoGPT, GPT-2 small/medium/large/XL)
  8.7  Loading pretrained weights (transformers required)
  8.10 Loss = NLL of next-token prediction

Run: python code/chapter08/gpt.py
"""

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Components from earlier chapters (inlined for self-containment) ──

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * d_model * 4 / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.W_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_head = config.n_embd // config.n_head

        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.W_o(out))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x


# ── 8.2 GPTConfig ─────────────────────────────────────────────

print("=" * 60)
print("8.2 — GPTConfig")
print("=" * 60)


@dataclass
class GPTConfig:
    block_size: int = 1024    # max context length
    vocab_size: int = 50257   # GPT-2's BPE vocab
    n_layer: int = 12         # number of blocks
    n_head: int = 12          # number of attention heads
    n_embd: int = 768         # embedding dimension
    dropout: float = 0.0      # dropout rate
    bias: bool = True         # use bias in Linears / LayerNorms


# ── 8.3 The GPT model ─────────────────────────────────────────

print("\n" + "=" * 60)
print("8.3 — GPT model")
print("=" * 60)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        # Init
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections (per GPT-2)
        for pn, p in self.named_parameters():
            if pn.endswith("W_o.weight") or pn.endswith("W_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gamma)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.transformer.wte(idx)            # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)            # [T, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                       # [B, T, vocab_size]

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
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── 8.6 Parameter count for the family ───────────────────────

print("\n" + "=" * 60)
print("8.6 — Parameter count for the GPT family")
print("=" * 60)

CONFIGS = {
    "nanoGPT (our SLM)": dict(n_layer=6,  n_head=6,  n_embd=384,  vocab_size=50257),
    "GPT-2 small":       dict(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257),
}

# Larger variants — uncomment to print (slow to build on CPU)
# "GPT-2 medium":      dict(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257),
# "GPT-2 large":       dict(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257),
# "GPT-2 XL":          dict(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257),

print(f"{'model':25}  {'n_layer':>8}  {'n_head':>7}  {'n_embd':>7}  {'params':>15}")
print(f"{'-----':25}  {'-------':>8}  {'------':>7}  {'------':>7}  {'------':>15}")
for name, kw in CONFIGS.items():
    cfg = GPTConfig(block_size=1024, **kw)
    m = GPT(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:25}  {kw['n_layer']:>8}  {kw['n_head']:>7}  "
          f"{kw['n_embd']:>7}  {n:>15,}")

# Reference: known parameter counts for the larger GPT-2 variants
print("\nReference (from GPT-2 paper, not computed here):")
print(f"  {'GPT-2 medium':25}  {24:>8}  {16:>7}  {1024:>7}  {354_000_000:>15,}")
print(f"  {'GPT-2 large':25}  {36:>8}  {20:>7}  {1280:>7}  {774_000_000:>15,}")
print(f"  {'GPT-2 XL':25}  {48:>8}  {25:>7}  {1600:>7}  {1_558_000_000:>15,}")


# ── 8.4 Forward pass demo ────────────────────────────────────

print("\n" + "=" * 60)
print("8.4 — Forward pass: shapes at every step")
print("=" * 60)

torch.manual_seed(0)
cfg = GPTConfig(block_size=128, vocab_size=1000, n_layer=2, n_head=4, n_embd=64)
model = GPT(cfg)
B, T = 2, 16
idx = torch.randint(0, 1000, (B, T))
targets = torch.randint(0, 1000, (B, T))

logits, loss = model(idx, targets)
print(f"Input ids:    {idx.shape}    (B={B}, T={T})")
print(f"Output logits: {logits.shape}  (B, T, vocab_size)")
print(f"Loss:         {loss.item():.3f}")
print(f"Expected initial loss: ~log(1000) = {math.log(1000):.3f}  (random predictions)")


# ── 8.5 Generation with temperature and top-k ────────────────

print("\n" + "=" * 60)
print("8.5 — Generation: temperature and top-k")
print("=" * 60)

# Train briefly so the model produces non-uniform distributions
print("Training briefly so the model has something to say...")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(300):
    batch = torch.randint(0, 1000, (8, 16))
    targets = torch.randint(0, 1000, (8, 16))
    _, loss = model(batch, targets)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Final training loss: {loss.item():.3f}")

# Try several generation settings
context = torch.tensor([[42, 17, 88, 4, 56]])   # arbitrary starting tokens
print(f"\nContext: {context[0].tolist()}")
for temp in [0.0, 0.5, 1.0]:
    for top_k in [None, 10]:
        torch.manual_seed(0)   # reproducible
        out = model.generate(context, max_new_tokens=10,
                             temperature=temp, top_k=top_k)
        suffix = out[0, context.size(1):].tolist()
        label = f"temp={temp}, top_k={top_k}"
        print(f"  {label:25} → {suffix}")


# ── 8.10 Loss = NLL of next token ────────────────────────────

print("\n" + "=" * 60)
print("8.10 — Loss is the negative log-likelihood of the next token")
print("=" * 60)

# Manual loss computation: same as F.cross_entropy
torch.manual_seed(0)
model = GPT(GPTConfig(block_size=64, vocab_size=100, n_layer=2, n_head=4, n_embd=32))
idx = torch.randint(0, 100, (1, 8))
logits, loss_pytorch = model(idx, targets=idx)
print(f"PyTorch cross-entropy loss: {loss_pytorch.item():.4f}")

# Manual: log-softmax of logits, then negative log of the prob at the correct index
log_probs = F.log_softmax(logits, dim=-1)
manual = -log_probs[0, torch.arange(8), idx[0]].mean()
print(f"Manual NLL:                 {manual.item():.4f}")
print(f"Match: {torch.allclose(loss_pytorch, manual, atol=1e-5)}")


# ── 8.7 Loading pretrained GPT-2 weights (optional) ──────────

print("\n" + "=" * 60)
print("8.7 — Loading pretrained GPT-2 weights (optional, may be slow)")
print("=" * 60)

# Set SKIP_GPT2=1 (or pass --no-gpt2) to skip the 500MB download
import os
import sys
SKIP_GPT2 = "--no-gpt2" in sys.argv or os.environ.get("SKIP_GPT2") == "1"

if SKIP_GPT2:
    print("SKIP_GPT2=1 set — skipping pretrained load.")
else:
    cfg = GPTConfig(block_size=1024, vocab_size=50257,
                    n_layer=12, n_head=12, n_embd=768, bias=True)
    model = GPT(cfg)
    try:
        model = from_pretrained(model, "gpt2")
    except Exception as e:
        print(f"Pretrained load failed: {e}")
        print("Continuing without it. The model is still built and ready to train.")

try:
    from transformers import GPT2LMHeadModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def from_pretrained(model, hf_model_name="gpt2"):
    """Load OpenAI's GPT-2 weights into our model.

    Note: HF's GPT-2 stores Q,K,V as one combined `c_attn` matrix and the FFN
    as one combined `c_fc` matrix. A full mapping requires splitting those
    matrices. nanoGPT has the canonical implementation:

        https://github.com/karpathy/nanoGPT/blob/master/model.py

    This stub shows the structure; for a working pretrained load, see nanoGPT.
    """
    if not HAS_TRANSFORMERS:
        print("`transformers` not installed — cannot load pretrained weights.")
        return model

    print(f"Loading HuggingFace {hf_model_name} (skeleton, not full mapping)...")
    gpt2 = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_state = gpt2.state_dict()
    our_state = model.state_dict()
    n_loaded = 0
    for k in our_state:
        if k in hf_state and our_state[k].shape == hf_state[k].shape:
            our_state[k].copy_(hf_state[k])
            n_loaded += 1
    print(f"  Loaded {n_loaded}/{len(our_state)} parameters with matching shape.")
    print(f"  Remaining keys (combined QKV/FFN) require the nanoGPT split logic.")
    return model


# ── 8.7b Speed test ──────────────────────────────────────────

print("\n" + "=" * 60)
print("8.7b — Speed test")
print("=" * 60)

cfg = GPTConfig(block_size=1024, vocab_size=50257,
                n_layer=6, n_head=6, n_embd=384, bias=False)
model = GPT(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: nanoGPT config, {n_params:,} parameters")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Warm up
for _ in range(3):
    x = torch.randint(0, 50257, (4, 1024), device=device)
    _, _ = model(x)
if device == "mps":
    torch.mps.synchronize()

t0 = time.time()
n_iters = 20
for _ in range(n_iters):
    x = torch.randint(0, 50257, (4, 1024), device=device)
    _, _ = model(x)
if device == "mps":
    torch.mps.synchronize()
t1 = time.time()

total_tokens = n_iters * 4 * 1024
print(f"\n{n_iters} forward passes at B=4, T=1024 on {device}:")
print(f"  {total_tokens:,} tokens in {(t1-t0)*1000:.0f}ms")
print(f"  {total_tokens / (t1-t0):.0f} tokens/second")
print(f"  {(t1-t0) / n_iters * 1000:.1f}ms per forward pass")


print("\nDone — the model is the same shape as GPT-2, just smaller and trained from scratch.")
print("       Next: train it on real text (Chapter 9).")