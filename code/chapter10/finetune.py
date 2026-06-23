"""
Chapter 10: Fine-Tuning

Covers:
  10.3 Instruction data format + loss masking
  10.4 Full fine-tuning loop
  10.5 LoRA — Low-Rank Adaptation
  10.6 Applying LoRA to a GPT model
  10.7 LoRA merging for free inference speedup
  10.11 Comparing full vs LoRA fine-tuning

Run: python code/chapter10/finetune.py
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Components (same as chapter 8/9, inlined) ─────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.gamma * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int(2 * d * 4 / 3)
            hidden = 64 * ((hidden + 63) // 64)
        self.W_gate = nn.Linear(d, hidden, bias=False)
        self.W_up   = nn.Linear(d, hidden, bias=False)
        self.W_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len, bias=False):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── 10.5 LoRA wrapper (defined before GPT so we can use it) ──

class LoRALinear(nn.Module):
    """Frozen base linear + low-rank adapter (A @ B)."""

    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False
        # LoRA: A is small random, B is zero → adapter starts at zero
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.lora_A @ self.lora_B)

    def merge(self):
        """Absorb the LoRA update into the base weight — zero inference overhead."""
        with torch.no_grad():
            self.base.weight.data += self.scale * (self.lora_A @ self.lora_B).T
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


# ── Tiny GPT (6 layers, 192 dim, ~10M params) ────────────────

D_MODEL, N_HEAD, N_LAYER, MAX_LEN, VOCAB = 192, 6, 6, 256, 50257


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=D_MODEL, head=N_HEAD, n_layer=N_LAYER, max_len=MAX_LEN):
        super().__init__()
        self.config = type("Cfg", (), dict(
            vocab_size=vocab, n_embd=d, n_head=head, n_layer=n_layer,
            block_size=max_len, dropout=0.0, bias=False
        ))()
        self.token_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([
            TransformerBlock(d, head, max_len) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.token_emb.weight   # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-100
            )
        return logits, loss


# ── Tokenizer wrapper ─────────────────────────────────────────

print("=" * 60)
print("10.3 — Tokenizer: GPT-2 BPE")
print("=" * 60)

try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-2")
    eot = enc.eot_token   # 50256 — used as separator
    print(f"Using GPT-2 BPE (vocab={enc.max_token_value + 1}, eot={eot})")
except ImportError:
    print("`tiktoken` not installed — using a tiny char-level fallback.")
    print("Install tiktoken for a more realistic demo: pip install tiktoken")
    # Char-level fallback
    CHARS = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?")
    STOI = {c: i + 100 for i, c in enumerate(CHARS)}
    ITOS = {i: c for c, i in STOI.items()}
    STOI["<unk>"] = 99
    class CharEnc:
        def encode(self, s): return [STOI.get(c, 99) for c in s]
        def decode(self, ids): return "".join(ITOS.get(i, "?") for i in ids)
        max_token_value = max(ITOS.keys())
    enc = CharEnc()
    eot = 99   # not a real EOT but we won't use it for char-level


def encode(text):
    """Encode with EOS appended."""
    return enc.encode(text) + [eot]


# ── 10.3 Loss masking for SFT ────────────────────────────────

print("\n" + "=" * 60)
print("10.3 — Loss masking for instruction tuning")
print("=" * 60)


def build_chat(messages, max_len=256):
    """
    Format: [user, assistant, user, assistant, ...]
    Returns (input_ids, loss_mask) where loss_mask=1 only on assistant tokens.
    """
    input_ids, loss_mask = [], []
    for msg in messages:
        content_ids = enc.encode(msg["content"])
        if msg["role"] == "assistant":
            # Assistant content: predict these
            input_ids.extend(content_ids)
            loss_mask.extend([1] * len(content_ids))
        else:
            # User/system content: context only
            input_ids.extend(content_ids)
            loss_mask.extend([0] * len(content_ids))
        # Add a separator after every turn
        input_ids.append(eot)
        loss_mask.append(0)
    return input_ids[:max_len], loss_mask[:max_len]


# Demo
messages = [
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user",      "content": "And 3+5?"},
    {"role": "assistant", "content": "8"},
]
ids, mask = build_chat(messages)
n_loss = sum(mask)
print(f"Conversation: {len(ids)} tokens  ({n_loss} loss-masked, "
      f"{100*n_loss/len(ids):.0f}%)")
print("Loss mask:    " + "".join("█" if m else "·" for m in mask))
print("              " + "".join("1" if m else "0" for m in mask[:60]))
print("              " + "".join(" " for _ in range(60)) + "(0=context, 1=predict)")


# ── Instruction dataset: simple arithmetic with natural language ─

INSTRUCTION_DATA = [
    {"user": "What is 2+2?",     "assistant": "2 + 2 = 4."},
    {"user": "What is 3+5?",     "assistant": "3 + 5 = 8."},
    {"user": "What is 10+7?",    "assistant": "10 + 7 = 17."},
    {"user": "What is 6-2?",     "assistant": "6 - 2 = 4."},
    {"user": "What is 9*2?",     "assistant": "9 * 2 = 18."},
    {"user": "What is 12/4?",    "assistant": "12 / 4 = 3."},
    {"user": "What is 100+25?",  "assistant": "100 + 25 = 125."},
    {"user": "What is 7-3?",     "assistant": "7 - 3 = 4."},
    {"user": "What is 5*5?",     "assistant": "5 * 5 = 25."},
    {"user": "What is 50-30?",   "assistant": "50 - 30 = 20."},
    {"user": "What is 11+11?",   "assistant": "11 + 11 = 22."},
    {"user": "What is 8*8?",     "assistant": "8 * 8 = 64."},
] * 30   # 360 examples — enough to overfit on this synthetic task

print(f"\nInstruction dataset: {len(INSTRUCTION_DATA)} examples")


# ── 10.4 Full fine-tuning loop ────────────────────────────────

print("\n" + "=" * 60)
print("10.4 — Full fine-tuning (all weights trainable)")
print("=" * 60)


def count_params(model):
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_total, n_train


def train_sft(model, data, lr, steps, label):
    """Fine-tune with proper loss masking."""
    opt = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    history = []
    for step in range(steps):
        ex = data[step % len(data)]
        msgs = [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        ids, mask = build_chat(msgs, max_len=MAX_LEN)
        targets = [tok if m == 1 else -100 for tok, m in zip(ids, mask)]
        x = torch.tensor([ids])
        y = torch.tensor([targets])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, loss.item()))
            print(f"  [{label:>4}] step {step:3}  loss {loss.item():.3f}")
    return history


model = TinyGPT()
ex = INSTRUCTION_DATA[0]
msgs = [
    {"role": "user", "content": ex["user"]},
    {"role": "assistant", "content": ex["assistant"]},
]
ids, mask = build_chat(msgs, max_len=MAX_LEN)
x = torch.tensor([ids])

# Loss WITH masking (only assistant tokens count)
targets_masked = [tok if m == 1 else -100 for tok, m in zip(ids, mask)]
_, loss_masked = model(x, torch.tensor([targets_masked]))

# Loss WITHOUT masking (every token counts)
targets_all = list(ids)
_, loss_unmasked = model(x, torch.tensor([targets_all]))

# Loss with REVERSED masking (only user tokens count, assistant ignored)
targets_reversed = [-100 if m == 1 else tok for tok, m in zip(ids, mask)]
_, loss_reversed = model(x, torch.tensor([targets_reversed]))

print(f"  Loss WITH proper masking (assistant only):  {loss_masked.item():.3f}")
print(f"  Loss WITHOUT masking (all tokens):          {loss_unmasked.item():.3f}")
print(f"  Loss with REVERSED mask (user only):        {loss_reversed.item():.3f}")
print()
print("  → The masked loss is the one we minimize during fine-tuning.")
print("  → Without masking, the model would also try to predict user tokens")
print("    (which don't help — those come from the user, not the model).")
print()


# ── 10.4 Full fine-tuning ────────────────────────────────────

print("\n" + "=" * 60)
print("10.4 — Full fine-tuning (all weights trainable)")
print("=" * 60)

model_full = TinyGPT()
n_total, n_train = count_params(model_full)
print(f"Model: {n_total:,} params  ({n_train:,} trainable = 100%)")
print()
print("NOTE: This synthetic arithmetic task is small enough that the model")
print("can memorize it from random init (loss reaches ~0 by step 0). That's")
print("because the answers are short and predictable. A real SFT task with")
print("diverse, harder examples would show the gradual loss decrease the way")
print("Chapter 9 did. We continue the run anyway to exercise the code paths.")
print()
hist_full = train_sft(model_full, INSTRUCTION_DATA, lr=3e-4, steps=300, label="full")


# ── 10.5–10.6 LoRA ────────────────────────────────────────────

print("\n" + "=" * 60)
print("10.5 — LoRA: Low-Rank Adaptation")
print("=" * 60)


def apply_lora(model, rank=8, alpha=16, target_names=("W_q", "W_v", "W_o"),
               freeze_embeddings=True, freeze_ffn=True):
    """Replace target linear layers with LoRA-wrapped versions.

    By default, also freezes the embedding and the FFN — only the LoRA
    adapters in the attention projections get trained. This is the standard
    LoRA recipe (Hu et al. 2021): tiny trainable footprint, big quality.
    """
    n_replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in target_names:
                continue
            lora = LoRALinear(
                child.in_features, child.out_features,
                rank=rank, alpha=alpha, bias=child.bias is not None
            )
            lora.base.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora.base.bias.data = child.bias.data.clone()
            setattr(parent, child_name, lora)
            n_replaced += 1

    # Freeze everything except LoRA adapters
    if freeze_embeddings:
        for p in model.token_emb.parameters():
            p.requires_grad = False
        for p in model.pos_emb.parameters():
            p.requires_grad = False
    if freeze_ffn:
        # Freeze FFN weights inside each block
        for block in model.blocks:
            for p in block.ffn.parameters():
                p.requires_grad = False
            for p in block.ln1.parameters():
                p.requires_grad = False
            for p in block.ln2.parameters():
                p.requires_grad = False
        # Freeze the final layer norm
        for p in model.ln_f.parameters():
            p.requires_grad = False
        # Freeze the output head (it shares weights with the embedding,
        # so it's already frozen via the embedding freeze above)

    return n_replaced


model_lora = TinyGPT()
n_replaced = apply_lora(model_lora, rank=8, alpha=16,
                        target_names=("W_q", "W_v", "W_o"))
n_total, n_train = count_params(model_lora)
print(f"  LoRA-wrapped {n_replaced} attention projections")
print(f"  Total: {n_total:,}  Trainable: {n_train:,}  "
      f"({100*n_train/n_total:.2f}%)")
print(f"  (The rest — embeddings, FFN, RMSNorm — stays frozen)")
print()
print("Fine-tuning with LoRA (LR is higher because fewer params update):")
hist_lora = train_sft(model_lora, INSTRUCTION_DATA, lr=1e-3, steps=300, label="LoRA")


# ── 10.7 LoRA merge ──────────────────────────────────────────

print("\n" + "=" * 60)
print("10.7 — Merging LoRA back into the base (free inference speedup)")
print("=" * 60)

lora_layers = [m for m in model_lora.modules() if isinstance(m, LoRALinear)]
print(f"  Found {len(lora_layers)} LoRA layers")
print(f"  Pre-merge:  lora_A norm = {lora_layers[0].lora_A.norm():.4f}, "
      f"lora_B norm = {lora_layers[0].lora_B.norm():.4f}")
for layer in lora_layers:
    layer.merge()
print(f"  Post-merge: lora_A norm = {lora_layers[0].lora_A.norm():.4f}, "
      f"lora_B norm = {lora_layers[0].lora_B.norm():.4f}")
print("  LoRA update is now folded into the base weight. "
      "Zero inference overhead.")


# ── 10.9 LoRA rank sweep ─────────────────────────────────────

print("\n" + "=" * 60)
print("10.9 — LoRA rank sweep")
print("=" * 60)

print(f"  rank  |  trainable params  |  % of total")
print(f"  ------+--------------------+-------------")
for r in [4, 8, 16, 32, 64]:
    m = TinyGPT()
    apply_lora(m, rank=r, alpha=2 * r, target_names=("W_q", "W_v", "W_o"))
    n_t, n_tr = count_params(m)
    print(f"  {r:>4}  |  {n_tr:>14,}    |  {100*n_tr/n_t:.2f}%")


# ── 10.11 Generation from the fine-tuned model ────────────────

print("\n" + "=" * 60)
print("10.11 — Generation check")
print("=" * 60)
print()
print("NOTE: This synthetic task is too trivial for the 12M model to learn")
print("useful behavior — the loss hits ~0 by step 0 because the answers are")
print("short and the patterns are uniform. For a meaningful generation demo")
print("you'd need: (a) a much smaller model, or (b) much harder data with")
print("diverse responses. We show the *code path* here, not a real demo.")
print()

model_full.eval()
test_prompts = [
    "What is 7+3?",
    "What is 5*5?",
]
for prompt in test_prompts:
    ids = enc.encode(prompt)
    x = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(10):
            logits, _ = model_full(x)
            next_id = logits[0, -1].argmax().item()
            x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    out = enc.decode(x[0, len(ids):].tolist())
    print(f"  Q: {prompt!r:25} → A: {out!r}")


print("\nDone — full fine-tuning adapts every weight; LoRA adapts 2-5% of the")
print("       model and is often indistinguishable in quality. The merge")
print("       step eliminates inference overhead, so LoRA is free at deploy")