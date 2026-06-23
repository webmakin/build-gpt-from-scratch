"""
Chapter 9: Training a GPT

Covers:
  9.2  Dataset loading and tokenization
  9.3  Train / validation split
  9.4  Random-window DataLoader
  9.5  AdamW optimizer
  9.6  Linear warmup + cosine decay LR schedule
  9.7  The full training loop with periodic evaluation
  9.8  Gradient clipping at norm 1.0
  9.10 Sampling from the trained model

Run: python code/chapter09/train.py
"""

import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── GPT model (compact version of chapter 8's GPT) ─────────────

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


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


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
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
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
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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


# ── 9.2 Dataset ───────────────────────────────────────────────

print("=" * 60)
print("9.2 — Dataset: Tiny Shakespeare")
print("=" * 60)

DATA_PATH = "datasets/tinyshakespeare.txt"
if not os.path.exists(DATA_PATH):
    os.makedirs("datasets", exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, DATA_PATH)

with open(DATA_PATH) as f:
    text = f.read()

# Tokenize
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode(text)
    print(f"Tokenizer: GPT-2 BPE ({enc.max_token_value + 1} vocab)")
except ImportError:
    print("`tiktoken` not installed; falling back to char-level tokenization")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    tokens = [stoi[ch] for ch in text]
    enc = type("E", (), {
        "encode": lambda s, tokens=tokens: [stoi[ch] for ch in s],
        "decode": lambda ids, itos=itos: "".join(itos[i] for i in ids),
    })()

data = torch.tensor(tokens, dtype=torch.long)
print(f"Total tokens: {len(data):,}")

# ── 9.3 Train / val split ────────────────────────────────────

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]
print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")


# ── 9.4 DataLoader ───────────────────────────────────────────

def get_batch(data, block_size, batch_size, device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size]     for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ── 9.6 LR schedule ──────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, peak_lr, min_lr_frac=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return peak_lr * min_lr_frac
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


# ── 9.7 Training config + loop ────────────────────────────────

@dataclass
class TrainingConfig:
    block_size: int = 64
    batch_size: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    max_steps: int = 500
    warmup_steps: int = 50
    peak_lr: float = 3e-4
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 10


@torch.no_grad()
def estimate_loss(model, train_data, val_data, tcfg, device):
    out = {}
    model.eval()
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, tcfg.block_size, tcfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, tcfg: TrainingConfig, device="cpu"):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tcfg.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=tcfg.weight_decay,
    )
    model.to(device)

    history = []   # (step, train_loss, val_loss)
    t_start = time.time()

    for step in range(tcfg.max_steps):
        # LR schedule
        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps,
                    tcfg.peak_lr, tcfg.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, tcfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        # 9.8 Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg.grad_clip)
        optimizer.step()
        sync()   # wait for GPU work to finish so timing is accurate

        # Periodic evaluation
        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            stats = estimate_loss(model, train_data, val_data, tcfg, device)
            elapsed = time.time() - t_start
            print(f"step {step:5}  lr {lr:.2e}  "
                  f"train {stats['train']:.3f}  val {stats['val']:.3f}  "
                  f"({elapsed:.0f}s)")
            history.append((step, stats['train'], stats['val']))

    print(f"\nTotal training time: {time.time() - t_start:.0f}s")
    return history


# ── Main ──────────────────────────────────────────────────────

device = "mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu"
print(f"Device: {device}")
# mps sync is needed for accurate timing on Apple Silicon
if device == "mps":
    def sync():
        torch.mps.synchronize()
else:
    sync = lambda: None

tcfg = TrainingConfig()
# Allow a fuller "demo" config via CLI flag for proper GPUs
if "--demo" in sys.argv:
    print("Using demo config (6L/6H/384d, 2000 steps) — needs a real GPU")
    tcfg = TrainingConfig(
        block_size=256, batch_size=12, n_layer=6, n_head=6, n_embd=384,
        max_steps=2000, warmup_steps=100, eval_interval=200,
    )
elif "--tiny" in sys.argv:
    # Default (no flag) is already small for CPU. --tiny is even smaller.
    tcfg = TrainingConfig(
        block_size=32, batch_size=32, n_layer=2, n_head=2, n_embd=64,
        max_steps=200, warmup_steps=20, eval_interval=50, eval_iters=5,
    )
print(f"\nTraining config: {tcfg}")

gcfg = GPTConfig(
    block_size=tcfg.block_size,
    vocab_size=50257,
    n_layer=tcfg.n_layer,
    n_head=tcfg.n_head,
    n_embd=tcfg.n_embd,
    dropout=0.0,
    bias=False,
)
model = GPT(gcfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters  "
      f"({n_params * 6 * tcfg.batch_size * tcfg.block_size / 1e12:.2f} TFLOPs/step)")

print("\n" + "=" * 60)
print("9.7 — Training")
print("=" * 60)

history = train(model, train_data, val_data, tcfg, device)


# ── 9.10 Sample from trained model ────────────────────────────

print("\n" + "=" * 60)
print("9.10 — Sample from the trained model")
print("=" * 60)

model.eval()
for prompt in ["ROMEO:", "JULIET:", "The king"]:
    ctx = torch.tensor([enc.encode(prompt)], device=device)
    out = model.generate(ctx, max_new_tokens=200,
                         temperature=0.8, top_k=200)
    text = enc.decode(out[0].tolist())
    print(f"\n--- prompt: {prompt!r} ---")
    print(text)
    print()


# ── 9.12 Loss curve plot ─────────────────────────────────────

print("=" * 60)
print("Loss curve (saved to /tmp/loss_curve.png)")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    train_losses = [h[1] for h in history]
    val_losses = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, "o-", label="train", color="steelblue")
    ax.plot(steps, val_losses, "s-", label="val", color="darkorange")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (NLL per token)")
    ax.set_title("Training loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/tmp/loss_curve.png", dpi=120)
    print("Saved /tmp/loss_curve.png")

    # If val_loss and train_loss diverged, the model overfit
    final_gap = val_losses[-1] - train_losses[-1]
    if final_gap > 0.2:
        print(f"\nFinal val-train gap: {final_gap:.3f}  "
              f"(>0.2 → significant overfitting on Tiny Shakespeare)")
    else:
        print(f"\nFinal val-train gap: {final_gap:.3f}  (healthy)")

except ImportError:
    print("matplotlib not available, skipping plot")


print("\nDone — you trained a GPT from scratch on real text.")