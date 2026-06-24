# Building GPT From Scratch

## Chapter 9

# Training a GPT

> *"Training is the moment the architecture stops being a sculpture and starts being a tool. Loss is the only thing that matters."*

---

## 9.1 What Training Actually Does

So far we have a `GPT` model that can forward-pass and generate, with randomly initialized weights. The output is uniform noise: the model has no idea what tokens are, let alone what to say next.

Training changes the weights so that `P(token_t | tokens_<t)` matches the actual distribution in some training corpus. We do this with **stochastic gradient descent**: repeatedly sample a batch of text, compute the loss (how wrong the model's predictions are), nudge every weight in the direction that would have made the loss smaller, repeat.

The loss is **negative log-likelihood of the actual next token**:

\\[
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(\text{token}_t \mid \text{tokens}_{1..t-1})
\\]

Minimizing this loss = maximizing the probability the model assigns to the training data. After enough updates, the model has learned enough of the structure of language to generate coherent text.

This chapter builds the full training pipeline: dataset loading, batching, the training loop itself, the optimizer, and the learning rate schedule. By the end you'll have a script that trains a GPT from scratch on real text.

---

## 9.2 The Dataset

A language model needs lots of text. For our SLM, we'll use something small enough to fit on a laptop — **Tiny Shakespeare** (~1MB of Shakespeare's plays) or any text file you have lying around. The principle is the same: feed it bytes, get out a model.

The most common text format for training is a single flat file:

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
...
```

We read the whole file into memory, tokenize it once (using the GPT-2 BPE tokenizer from Chapter 3), and store the resulting token IDs as a single 1D tensor. For Tiny Shakespeare this is roughly 300,000 tokens. For real training you want millions to billions.

```python
import tiktoken
import torch

# Tiny Shakespeare (auto-download if not present)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = ...   # read from file or download

# Tokenize once
enc = tiktoken.encoding_for_model("gpt2")
tokens = enc.encode(text)
data = torch.tensor(tokens, dtype=torch.long)   # [N], one giant 1D tensor

print(f"Total tokens: {len(data):,}")
# Total tokens: 338,025
```

The `data` tensor is the entire training set. We never hold a separate "list of sequences" — every batch is a different random window into this 1D tensor.

---

## 9.3 Train / Validation Split

To know whether the model is generalizing (not just memorizing), we hold out a fraction of the data for validation. Standard split: 90% train, 10% val.

```python
n = len(data)
train_data = data[: int(n * 0.9)]
val_data   = data[int(n * 0.9):]
print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")
# Train: 304,222  Val: 33,803
```

**Important:** the split is done *before* tokenization or *after*, but it must be done at the document level if you have multiple documents. For a single file, the split is just a position-based partition.

---

## 9.4 The DataLoader

A "batch" in language modeling is just `B` random windows of length `T` (the block size) into the 1D token stream. For each window, the input is the first `T` tokens and the target is the same window shifted by one position — the model must predict each token from the ones before it.

```python
def get_batch(data, block_size, batch_size):
    """Sample B random windows of length block_size+1 from data."""
    # Random starting positions
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    # Build the input/target pairs
    x = torch.stack([data[i : i + block_size]     for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


x, y = get_batch(train_data, block_size=64, batch_size=4)
print(f"x: {x.shape}  y: {y.shape}")
# x: torch.Size([4, 64])  y: torch.Size([4, 64])
# Each row of y is the same as x, shifted by one token to the right
```

That's it. No padding, no shuffling, no fancy collate function. Just random windows.

**Why random windows instead of sequential batches?** Two reasons:
1. **Variance reduction.** Each batch sees a different random sample of the data, which gives noisier but more representative gradient estimates.
2. **Implicit shuffling.** Sequential batches would mean the model always sees positions `[0:T], [T:2T], [2T:3T], ...`, which biases training toward local patterns.

The cost: each token is seen in many different contexts across training (different windows), which is actually a feature. It makes the data effectively larger.

---

## 9.5 The Optimizer — AdamW

Gradient descent updates each weight as `w ← w - lr * ∂L/∂w`. **AdamW** (Adam with decoupled weight decay) is the standard choice for transformer training. It maintains a per-parameter running estimate of the gradient's mean and variance, then applies the update with a small "decoupled" decay on the weights themselves.

```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,            # learning rate
    betas=(0.9, 0.95),  # exponential moving average coefficients
    weight_decay=0.1,   # decoupled weight decay
)
```

**Why these values?**
- `lr=3e-4` is the **nanoGPT default** and works well for 10M–1B parameter models. Smaller models can use higher LR; larger models should use lower.
- `betas=(0.9, 0.95)` is more aggressive than the default `(0.9, 0.999)` for transformers. The higher second beta means the variance estimate adapts faster, which helps in the early training steps.
- `weight_decay=0.1` applies a small decay that keeps the weights from growing too large. This is the "decoupled" part — it doesn't get mixed with the gradient, just applied to the weights directly.

**Decoupled weight decay** is a subtle but important detail. Naive L2 regularization adds `weight_decay * w` to the gradient, which gets scaled by Adam's adaptive learning rate. Decoupled weight decay applies `lr * weight_decay * w` as a separate step, *not* scaled by Adam. The result: weight decay has a consistent effect regardless of the gradient's magnitude.

---

## 9.6 The Learning Rate Schedule

Constant learning rate is a mistake. At the start of training, the model is far from a good solution and can take large steps. Later, large steps would cause oscillation. The standard schedule is:

1. **Linear warmup** from 0 to peak LR over the first `warmup_steps` steps
2. **Cosine decay** from peak LR to ~10% of peak over the rest of training

```python
import math

def get_lr(step, warmup_steps, max_steps, peak_lr, min_lr_frac=0.1):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return peak_lr * min_lr_frac
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * decay)


# Set the LR for this step
lr = get_lr(step=100, warmup_steps=100, max_steps=1000, peak_lr=3e-4)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

Why warmup? At initialization, the model's activations and gradients are at random magnitudes. Taking a large step from this random starting point can immediately destabilize training. Warmup lets the model find a sane region of parameter space before taking large steps.

Why cosine? The decay is smooth at the boundaries (zero derivative at step=max_steps) and spends more time at low LR where the model is fine-tuning.

**Total budget.** The "max_steps" decision determines how long you train. A rule of thumb for transformers:

```
chinchilla-optimal tokens = 20 × parameters
```

For our 30M parameter SLM, that's 600M tokens. Tiny Shakespeare has 338K — about 1800× too small. The model will overfit hard. For a real training run, use a larger corpus (OpenWebText, The Pile, SlimPajama, etc.).

---

## 9.7 The Training Loop

Putting it all together:

```python
import time
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    block_size: int = 256
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    max_steps: int = 2000
    warmup_steps: int = 100
    peak_lr: float = 3e-4
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    eval_interval: int = 200
    eval_iters: int = 20


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """Compute mean loss over a few batches of train and val data."""
    out = {}
    model.eval()
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.block_size, config.batch_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, config: TrainingConfig, device="cpu"):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    model.to(device)
    losses = []

    for step in range(config.max_steps):
        # LR schedule
        lr = get_lr(step, config.warmup_steps, config.max_steps,
                    config.peak_lr, config.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, config.block_size, config.batch_size)
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (the "gc" in some training scripts)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic evaluation
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            stats = estimate_loss(model, train_data, val_data, config, device)
            print(f"step {step:5}  lr {lr:.2e}  "
                  f"train {stats['train']:.3f}  val {stats['val']:.3f}")
            losses.append((step, stats['train'], stats['val']))

    return losses
```

That's a complete training script. About 50 lines. Run it for a few minutes and you'll have a trained GPT.

---

## 9.8 Gradient Clipping

The line `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is doing important work. Without it, the occasional "spike" in gradient magnitude (a single batch where the loss is much higher than average) can send the parameters flying to a region of weight space from which they never recover.

`clip_grad_norm_` rescales the gradient in-place so that its total L2 norm is at most `max_norm`. If the norm is below the threshold, no change. If above, the entire gradient is scaled down proportionally. This is **not** the same as `max_norm` per-parameter — it's a global norm constraint.

`max_norm=1.0` is the standard value for transformer training. Some papers use higher (5.0) or skip it entirely; for SLM training it's a no-brainer to include.

---

## 9.9 What "Loss" Means

After training, your model has a `val_loss` around 1.5 for Tiny Shakespeare after a few thousand steps with the full 30M model. A small (~7M parameter) model trained for 500 steps reaches `val_loss ≈ 5.7` — a useful demonstration but nowhere near coherent text. The full 30M model needs more steps and more parameters to get below 2.0. What does that number mean?

Loss is the negative log-likelihood per token:

\\[
\text{val\_loss} = -\frac{1}{T} \sum_{t=1}^{T} \log P(\text{actual\_token}_t \mid \text{context})
\\]

A few reference values:

| val_loss | Interpretation |
|---|---|
| 11.0 | Uniform over 50k vocab (log(50000)) — random model |
| 7.0 | Knows the most common words but nothing about structure |
| 5.0 | Recognizes word-level patterns, can complete "the cat ___" |
| 3.0 | Generates mostly coherent text with occasional garbled phrases |
| 2.0 | Fluent output, mostly grammatical, sometimes makes sense |
| 1.5 | Looks like it could have come from the training set |
| 1.0 | Memorizing — val_loss keeps dropping while train_loss is much lower |

The model can drive `train_loss` arbitrarily low by memorizing the training set. The interesting curve is **val_loss**: when it stops going down, the model has extracted as much generalizable signal as it can from this much data with this much capacity.

If `val_loss` plateaus and `train_loss` keeps dropping, you have **overfitting**. The fixes are: more data, smaller model, more regularization (dropout, weight decay), or earlier stopping.

---

## 9.10 Sampling After Training

After training, the model has weights. To use them:

```python
model.eval()
context = torch.tensor([enc.encode("ROMEO:")], device=device)   # [1, T]
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=200)
print(enc.decode(generated[0].tolist()))
```

A model trained on Tiny Shakespeare for 2000 steps will produce text that looks like Shakespeare but doesn't make sense:

```
ROMEO: I am a man, that hath no such thing;
And though I be a man to be so fair,
I will not be the prince of all the world.

JULIET:
What is thy name? I have a son of Greece,
And the son of the world, that I have seen
In the bosom of the sea, and I will die.
```

That's not Shakespeare. But it *is* a language model. It has learned:
- That speeches start with a name in capital letters followed by a colon
- That lines are short, often with "I" or "thou" or "and"
- That some words (Romeo, Juliet, man, world) co-occur
- That lines often end with periods or commas

To get actual coherent text, you need (a) much more data, (b) a much larger model, and (c) much longer training. Tiny Shakespeare is a *pedagogical* dataset — it teaches you the loop, not the destination.

---

## 9.11 Time and Compute

A 30M model on a small dataset trains in minutes. A 7B model on a trillion tokens takes weeks on hundreds of GPUs. The scaling is brutal but predictable: total training compute is roughly `6 × N × T` FLOPs, where `N` is parameters and `T` is tokens.

For our SLM with `N = 30M, T = 304K` (Tiny Shakespeare train split):

```
FLOPs ≈ 6 × 30e6 × 304e3 = 5.5e13  (55 TFLOPs)
```

A modern GPU (RTX 4090, A100) does ~150 TFLOPs in fp16. So one full epoch of Tiny Shakespeare is **a few seconds** on a single GPU. With our 2000-step config, training takes a couple of minutes.

Real training (10B+ tokens, 1B+ parameters) is 10⁶× more compute — that's why frontier model training costs millions of dollars in compute.

---

## 9.12 Reading the Loss Curve

When you train, you'll see something like this:

```
step     0  lr 0.00e+00  train 10.95  val 10.95     ← start, near log(50000)
step   200  lr 3.00e-04  train  4.21  val  4.23     ← fast learning
step   400  lr 2.93e-04  train  3.18  val  3.25     ← continued
step   600  lr 2.78e-04  train  2.71  val  2.83     ← slowing
step   800  lr 2.55e-04  train  2.41  val  2.61     ← val gap appears
step  1000  lr 2.26e-04  train  2.20  val  2.48     ← overfitting begins
step  1500  lr 1.51e-04  train  1.85  val  2.35     ← train keeps dropping
step  2000  lr 0.30e-04  train  1.65  val  2.30     ← val plateau
```

Reading this:
- `train ≈ val` early on: model is generalizing
- `val` curve flattens while `train` keeps dropping: **overfitting**
- The `val` value at convergence tells you what the model knows

If you want a model that *learns more*, you need either more parameters, more data, or both. Tiny Shakespeare is a small dataset — for a 30M model, it will overfit in a few hundred steps. Real datasets (10B+ tokens) keep both curves dropping for much longer.

---

## 9.13 What Comes Next

After training, the model is a "base model" — it knows language structure but doesn't follow instructions. To make a chat model like ChatGPT, you need to fine-tune it on instruction-following data (Chapter 10), and then optionally RLHF it (Chapter 10, briefly). For inference at scale, you need KV caching, quantization, and serving infrastructure (Chapter 11).

But the model you train in this chapter is a *real* language model. It can be saved, loaded, fine-tuned, distilled, exported, and deployed. The hard part is done.

---

## Chapter Summary

- A "batch" in language modeling is `B` random windows of length `T` from a 1D token stream.
- **AdamW** with `lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1` is the standard optimizer.
- **LR schedule**: linear warmup (typically 5-10% of total steps) + cosine decay to 10% of peak.
- **Gradient clipping** at norm 1.0 prevents training spikes from destabilizing the run.
- **Loss** is the negative log-likelihood per token. `val_loss` plateauing is the signal that training is done.
- **Overfitting** is the most common failure mode: `val_loss` flat while `train_loss` falls. Fixes: more data, smaller model, more regularization.
- Tiny Shakespeare is a teaching dataset, not a training dataset. Real models need millions to billions of tokens.

In Chapter 10, we take a pretrained model and adapt it to do something specific — instruction-following, classification, code generation. That's **fine-tuning**.

---

## Exercises

1. **Loss curve plotting.** Add matplotlib plotting of the train/val loss curve. Where does val_loss plateau?
2. **Larger model.** Try `n_embd=512, n_layer=8` on the same data. Does it overfit faster or slower? (Faster — more parameters, same data.)
3. **Longer training.** Run for 10,000 steps with the same model. Does val_loss keep improving?
4. **Dropout.** Set `dropout=0.1` and retrain. Does the val_loss plateau at a higher or lower value?
5. **Different LR.** Try `peak_lr ∈ {1e-3, 3e-4, 1e-4}`. Which converges fastest? Which has the lowest final val_loss?
6. **Sampling parameters.** Generate from a trained model with `temperature ∈ {0.0, 0.5, 0.8, 1.2}`. How do the outputs differ in style?
7. **Different data.** Replace Tiny Shakespeare with a Python source file. What kind of structure does the model learn? Can it complete a function?
8. **GPU training.** Move the training loop to an MPS or CUDA device. Measure the speedup.

The full training script lives in `code/chapter09/train.py` — it downloads Tiny Shakespeare automatically, trains the nanoGPT config for 2000 steps, and samples from the result. Run it, watch the loss go down, then sample from your trained model.