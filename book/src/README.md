# Building GPT From Scratch

> **The Complete Engineering Guide to Building a Small Language Model (SLM)**
>
> From raw text to a production-ready GPT implementation — every line of code explained.

This book teaches you how to **build a GPT-style decoder-only transformer from first principles** in PyTorch. No black boxes, no `nn.MultiheadAttention(...)` shortcuts — every tensor, every matrix multiplication, every gradient is written and explained.

By the end, you'll have built a working Small Language Model with:

- A Byte Pair Encoding (BPE) tokenizer
- Multi-Head Self Attention
- RoPE positional embeddings
- RMSNorm & SwiGLU
- A complete training pipeline
- Mixed precision and gradient checkpointing
- And a chat interface to talk to your model

## How to read this book

Each chapter follows the same structure:

```
Problem
  ↓
History
  ↓
Mathematics
  ↓
Visualization
  ↓
Implementation
  ↓
Optimization
  ↓
Production Version
  ↓
Exercises
```

Code lives in the `code/` directory of the repository, organized by chapter. Run any chapter's code with `python code/chapterNN/...`.

## Companion repository

All source code, datasets, and exercises live at:
**[github.com/webmakin/build-gpt-from-scratch](https://github.com/webmakin/build-gpt-from-scratch)**

Issues and PRs are welcome — especially fixes to math/code errors, additional exercises, and diagrams.

Let's build GPT from scratch.
