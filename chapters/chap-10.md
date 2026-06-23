# Building GPT From Scratch

## Chapter 10

# Fine-Tuning

> *"Pretraining teaches a model to complete text. Fine-tuning teaches it to do what you want."*

---

## 10.1 Why Fine-Tune?

The model from Chapter 9 is a "base model." It has learned the statistical structure of language — what words follow what, how sentences are structured, the rhythm of prose. But it has no notion of:

- **Following instructions** ("Translate this to French")
- **Answering questions** ("What is the capital of France?")
- **Refusing harmful requests** ("I can't help with that")
- **Adopting a persona** ("You are a helpful assistant")

A base model prompted with "What is the capital of France?" will continue with more text in similar style — maybe "What is the capital of Italy? What is the capital of..." It answers the question by *extending the prompt*, not by being helpful.

Fine-tuning takes a base model and trains it on a curated dataset of (instruction, response) pairs. After a few hundred steps, the model learns to:

1. **Recognize the format** of an instruction (a question, a command, a request)
2. **Produce the corresponding response** in the same style as the training data
3. **Stop at the right time** (the end of the response, not at the end of an imagined continuation)

This is what turns a base model into a "chat model" or an "instruction-following assistant" — the GPT of GPT-3, the "Chat" in ChatGPT, the difference between a sentence completer and a conversational agent.

---

## 10.2 Three Eras of Fine-Tuning

The field has gone through three distinct approaches:

| Era | Method | Cost | Data required |
|---|---|---|---|
| 2017–2020 | Full fine-tuning of all weights | High | 10K–100K examples |
| 2021–2023 | LoRA / Adapters (freeze base, train small matrices) | Medium | 1K–10K examples |
| 2023+ | QLoRA (4-bit base + LoRA adapters) | Low | 1K–10K examples |

**Full fine-tuning** retrains every weight of the model. For a 7B parameter model that's 7B × 4 bytes = 28GB of optimizer state alone (AdamW needs 2× the weights in momentum and variance). Add gradients and activations, and a single training run needs 80+ GB of GPU memory. Effective, but expensive.

**LoRA (Low-Rank Adaptation)** freezes the base model and adds small trainable matrices to each linear layer. Original weights stay untouched; the model adapts by mixing in low-rank updates. ~1% of the original parameter count, ~1% of the memory. This is the sweet spot for most use cases.

**QLoRA** combines 4-bit quantization of the base model with LoRA adapters. The base model takes 4× less memory (7B in 4-bit is 3.5GB instead of 14GB), and the LoRA adapters train as usual. Lets you fine-tune 70B models on a single 24GB consumer GPU.

This chapter implements all three.

---

## 10.3 Instruction Data Format

A "chat" example is a list of turns. The standard format (used by ChatML, OpenAI, Llama 2 Chat, etc.) is:

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
And what's the population?<|im_end|>
<|im_start|>assistant
```

Each turn has a role (`system`, `user`, `assistant`) and content. The model is trained to predict the assistant's response given the prior conversation.

In code, we represent this as a list of dicts:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]
```

The training target is just the assistant turns — the model is told to predict only the assistant's responses, not the user's. The user turns are part of the *input* (the context the model sees) but they're masked out of the loss.

This is a key implementation detail. Let me make it explicit:

```python
def tokenize_with_loss_mask(messages, tokenizer, max_length):
    """
    Tokenize a conversation and return input_ids + a loss mask
    (1 for tokens the model should predict, 0 for context tokens).
    """
    input_ids = []
    loss_mask = []

    for msg in messages:
        # Format: <|im_start|>role\ncontent<|im_end|>\n
        header = f"<|im_start|>{msg['role']}\n"
        body = f"{msg['content']}<|im_end|>\n"

        header_ids = tokenizer.encode(header)
        body_ids = tokenizer.encode(body)

        input_ids.extend(header_ids)
        # Header is context, no loss
        loss_mask.extend([0] * len(header_ids))

        input_ids.extend(body_ids)
        if msg["role"] == "assistant":
            loss_mask.extend([1] * len(body_ids))
        else:
            # User and system are context
            loss_mask.extend([0] * len(body_ids))

    return input_ids[:max_length], loss_mask[:max_length]
```

The `loss_mask` becomes the `ignore_index` argument to `cross_entropy`: we set target to `-100` wherever the mask is 0. The loss is computed only on tokens the model should learn to predict.

---

## 10.4 Full Fine-Tuning

The simplest case. Load the base model, load instruction data, train all the weights. Two changes from the pretraining loop:

1. **Custom loss mask** — only compute loss on assistant tokens
2. **Smaller learning rate** — 1e-5 to 5e-5 for instruction tuning (vs 3e-4 for pretraining)

The full loop looks like:

```python
def finetune(model, instruction_data, lr=2e-5, max_steps=1000):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for step in range(max_steps):
        messages = sample(instruction_data)              # (input, target) pair
        input_ids, loss_mask = tokenize_with_loss_mask(messages, tokenizer, max_length=1024)

        # Build targets: same as input_ids, but -100 where loss_mask is 0
        targets = torch.tensor(input_ids).clone()
        targets[loss_mask == 0] = -100

        x = torch.tensor([input_ids])
        y = torch.tensor([targets])
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

That's the entire fine-tuning loop. The training script from Chapter 9 with two changes (loss mask, lower LR) is all you need.

---

## 10.5 LoRA — Low-Rank Adaptation

The insight: when fine-tuning, the weight updates `ΔW` are often **low-rank** — they have a few important directions and the rest is noise. So instead of learning a full `ΔW ∈ ℝ^{d×d}`, we learn `ΔW = AB` where `A ∈ ℝ^{d×r}` and `B ∈ ℝ^{r×d}` with rank `r << d`.

For `d=512, r=8`, the original layer has `512² = 262K` parameters; the LoRA adapter has `512 × 8 + 8 × 512 = 8K` parameters — 32× smaller.

```python
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter."""

    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        # Frozen base
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False
        # LoRA adapter
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank   # LoRA scaling

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A) @ self.lora_B
        return base_out + self.scale * lora_out
```

The base is frozen, the LoRA matrices are trainable. After training, you can either:

- **Keep them inline** for inference: `out = base(x) + scale * (x @ A @ B)`. Slight compute overhead.
- **Merge them** into the base: `W_new = W + scale * A @ B`. Then `lora_out` is computed "for free" inside the base matmul. Same model, no inference overhead.

```python
# Merge LoRA into base (for inference)
def merge_lora(layer: LoRALinear):
    layer.base.weight.data += layer.scale * (layer.lora_A @ layer.lora_B).T
    # Now you can ignore lora_A, lora_B
```

Initialization matters: `lora_A` is initialized small random, `lora_B` is initialized to zero. So at the start of training, `lora_out = 0` and the model behaves identically to the base. Training grows the LoRA from zero.

---

## 10.6 Applying LoRA to a GPT

Replacing the linear layers in attention and FFN with LoRA variants:

```python
def apply_lora(model, rank=8, alpha=16, target_modules=("W_q", "W_v", "W_o", "W_gate", "W_up", "W_down")):
    """Replace target linear layers with LoRA-wrapped versions."""
    for name, module in list(model.named_modules()):
        # Check if this is a target module
        if not any(name.endswith(f".{t}") for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Create a LoRA wrapper, copy the base weights
        lora = LoRALinear(
            module.in_features, module.out_features,
            rank=rank, alpha=alpha, bias=module.bias is not None
        )
        lora.base.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora.base.bias.data = module.bias.data.clone()

        # Replace in the parent module
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, lora)

    return model


# Apply LoRA, count trainable parameters
model = apply_lora(model, rank=8, alpha=16)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_trainable:,} / {n_total:,}  "
      f"({100 * n_trainable / n_total:.2f}%)")
```

For a 7B model, this gives ~1% trainable parameters (~70M for full attention+FFN LoRA on every layer). Training is much faster and uses much less GPU memory.

---

## 10.7 QLoRA — 4-bit Base + LoRA Adapters

QLoRA is the production standard for consumer-GPU fine-tuning of 70B-class models. The trick: quantize the base model to 4-bit NF4 (NormalFloat 4), then add LoRA adapters on top. The adapters stay in fp16/bf16, the base is 4-bit.

The full pipeline:

1. **Quantize base** to NF4: 4-bit per weight instead of 16-bit. ~4× memory savings.
2. **Dequantize on-the-fly** during forward: when a layer needs to compute, dequantize its weights to fp16, do the matmul, throw away. The dequantization is cheap and the matmul still happens in higher precision.
3. **Add LoRA adapters** that operate in fp16: the low-rank updates are tiny, so they don't blow up the memory budget.
4. **Train** normally with backprop; only the LoRA parameters get gradient updates.

In practice, QLoRA needs the `bitsandbytes` library:

```bash
pip install bitsandbytes
```

```python
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    quantization_config=bnb.QuantizationConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
# Add LoRA adapters (peft library does this in one line)
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

For our from-scratch GPT, QLoRA would mean:

- **Storage**: 4-bit weights for the base (~7.5MB per 15M parameters)
- **Forward**: dequantize → matmul in fp16 → discard
- **Backward**: only LoRA parameters get gradients

The compute cost of dequantization is small compared to the matmul, so the training is essentially as fast as full fp16 training — but with ~4× less memory.

We won't implement QLoRA from scratch in this book (the bitsandbytes integration is non-trivial), but the principle is clear: quantize the base, add LoRA adapters, train only the adapters.

---

## 10.8 Data Sources for Fine-Tuning

The hardest part of fine-tuning is the data. A few canonical datasets:

- **Alpaca** (52K examples) — Self-instruct from GPT-3.5. Low quality but a starting point.
- **Dolly** (15K examples) — Human-written by Databricks staff. Higher quality.
- **OpenAssistant** (88K examples) — Crowd-sourced conversations. Diverse.
- **LMSYS-Chat-1M** (1M examples) — Real conversations from Chatbot Arena. The best for chat training.
- **HH-RLHF** (human preference) — Anthropic's helpful-harmless dataset with both chosen and rejected responses. Used for RLHF, not SFT.

For domain-specific fine-tuning, you want a few thousand high-quality examples. Quantity is less important than quality: 5,000 carefully written examples beat 100,000 noisy ones.

**Building your own dataset** is the highest-leverage approach. If you have 1,000 examples of (question, ideal-answer) pairs from your domain, fine-tuning on them is more useful than any of the public datasets.

---

## 10.9 Hyperparameters for Fine-Tuning

The defaults that work for most cases:

| Parameter | Pretraining | SFT (instruction tuning) | LoRA SFT |
|---|---|---|---|
| Learning rate | 3e-4 | 2e-5 to 5e-5 | 1e-4 to 3e-4 |
| Batch size | 32–128 | 4–32 | 4–16 |
| Warmup steps | 100–1000 | 50–200 | 50–100 |
| Weight decay | 0.1 | 0.01 | 0.01 |
| Epochs | 1 (large data) | 3 (small data) | 3–5 |
| Sequence length | 1024–4096 | 512–2048 | 512–2048 |

SFT uses a much smaller LR than pretraining because we're not moving the model far from its pretrained state. A high LR "forgets" the pretraining.

For LoRA, the effective LR is higher because each step only updates a small percentage of the parameters. The scaling factor `alpha / r` also matters: `alpha = 2 * r` is a common default.

---

## 10.10 Common Pitfalls

**Catastrophic forgetting.** Fine-tune too long, and the model loses its general language ability. Mitigation: lower LR, fewer epochs, mix pretraining data in.

**Mode collapse.** The model becomes very good at one narrow task type and loses diversity. Mitigation: diverse training data, lower LR.

**Overfitting to format.** The model memorizes the exact response template instead of learning the content. Mitigation: vary your templates, use a held-out validation set.

**Reward hacking (in RLHF).** If the reward model has flaws, the policy can exploit them. Mitigation: careful reward model design, KL penalty to the base.

**LoRA rank too low.** If `r=2` isn't enough capacity, the model can't learn the task. Try `r ∈ {8, 16, 32, 64}` and pick the smallest that achieves good validation loss.

**LoRA rank too high.** Rank too high = LoRA is just full fine-tuning. Loses the memory savings. Start small and grow.

---

## 10.11 Evaluating Fine-Tuned Models

A fine-tuned model is only as good as its evaluation. Three common approaches:

**1. Held-out validation set.** Hold out 10% of your data, compute loss. Lower loss on the held-out set = the model is generalizing. Easy to measure, but loss is a weak proxy for quality.

**2. LLM-as-judge.** Use GPT-4 (or another strong LLM) to compare outputs from your model against reference answers. Gives quality scores correlated with human judgment, but expensive and has its own biases.

**3. Human evaluation.** The gold standard. Show humans pairs of (input, output) from two models and ask "which is better?" Slow, expensive, but the only one that doesn't have circular dependencies.

For instruction tuning, **format compliance** is the easiest thing to measure: does the model follow the expected response structure? For **content quality**, you need either LLM-as-judge or humans.

---

## 10.12 What Comes Next

After instruction tuning, the model can follow instructions. But it might not follow them *well*. To get from "kind of follows instructions" to "follows them like ChatGPT" requires:

- **RLHF** (Reinforcement Learning from Human Feedback) — train a reward model on human preferences, then PPO against the reward. Expensive but powerful.
- **DPO** (Direct Preference Optimization) — skip the reward model, train directly on preference pairs. Simpler, often equivalent.
- **Constitutional AI** — use a strong LLM to generate critiques and revisions, fine-tune on the revised outputs.

These are out of scope for the from-scratch book, but they're the next steps. For most applications, a well-done SFT is enough.

---

## Chapter Summary

- **Fine-tuning** adapts a pretrained base model to follow instructions or specialize in a domain.
- **Full fine-tuning** retrains all weights. Effective but expensive.
- **LoRA** freezes the base and adds low-rank adapter matrices. ~1% of parameters, ~1% of memory.
- **QLoRA** adds 4-bit quantization to LoRA. Lets you fine-tune 70B models on a 24GB GPU.
- **Loss masking** is critical: only compute loss on assistant tokens, not on user input.
- **Data quality > data quantity** for fine-tuning. 1,000 good examples beat 100,000 noisy ones.
- SFT uses a much lower LR than pretraining (1e-5 vs 3e-4) to avoid catastrophic forgetting.

In Chapter 11, we look at **inference optimization** — making the trained model run faster and use less memory at generation time.

---

## Exercises

1. **LoRA vs full fine-tuning.** Train one model with full fine-tuning and one with LoRA on the same 100 examples. Compare final loss and parameter count.
2. **LoRA rank sweep.** Try `r ∈ {4, 8, 16, 32, 64}`. At what rank do you stop seeing quality improvements?
3. **Loss masking.** Without loss masking, what does the model learn? Train a model with no mask for 200 steps and compare generations to a model with proper masking.
4. **Data quality experiment.** Take Alpaca, split it into "high quality" (manual review) and "low quality" subsets. Fine-tune on each. Compare results.
5. **Format compliance.** Build a small set of (instruction, response) pairs in a specific JSON format. Fine-tune and check if the model produces valid JSON.
6. **LoRA on different target modules.** Try `target_modules = ["W_q", "W_v"]` vs `["W_q", "W_k", "W_v", "W_o"]` vs all linear layers. Where do LoRA adapters help most?
7. **Save and reload LoRA.** After training, save just the LoRA adapter (a few MB). Load it back into the base model and verify generations match.

The full implementation lives in `code/chapter10/finetune.py` — a complete SFT + LoRA pipeline that you can run on the trained model from Chapter 9.