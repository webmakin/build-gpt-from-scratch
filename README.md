# 🚀 Building GPT From Scratch

> **The Complete Engineering Guide to Building a Small Language Model (SLM)**
>
> From raw text to a production-ready GPT implementation—every line of code explained.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)]()

---

## 📖 About

Most tutorials teach **how to use** GPT models.

This project teaches you **how to build one from scratch.**

By the end of this book, you'll understand every component of a modern decoder-only transformer—from tokenization and embeddings to multi-GPU training, inference optimization, and deployment.

This isn't just another AI tutorial.

It's a **complete engineering handbook** for building production-quality Small Language Models (SLMs).

Everything is implemented from first principles using **PyTorch**, with minimal reliance on external libraries.

---

# 🎯 What You'll Build

By the end of this project, you'll have built:

- ✅ A Byte Pair Encoding (BPE) tokenizer
- ✅ A GPT-style decoder-only Transformer
- ✅ Multi-Head Self Attention
- ✅ RoPE Positional Embeddings
- ✅ RMSNorm & SwiGLU
- ✅ Training Pipeline
- ✅ Mixed Precision Training
- ✅ Distributed Training (DDP/FSDP)
- ✅ LoRA & QLoRA Fine-tuning
- ✅ GGUF Export
- ✅ KV Cache
- ✅ Streaming Text Generation
- ✅ FastAPI Inference Server
- ✅ Docker Deployment
- ✅ Kubernetes Deployment
- ✅ Complete ChatGPT-style Web Interface

---

# 📚 Book Roadmap

## Part I — Foundations

- Introduction to Language Models
- Mathematics for Machine Learning
- Linear Algebra
- Calculus & Backpropagation
- Probability
- Gradient Descent

---

## Part II — Text Processing

- Unicode
- UTF-8
- Tokenization
- Byte Pair Encoding (BPE)
- Vocabulary Construction

---

## Part III — Neural Networks

- Embeddings
- Positional Encoding
- Rotary Embeddings (RoPE)
- Feed Forward Networks
- Layer Normalization

---

## Part IV — Transformers

- Attention
- Self Attention
- Multi-Head Attention
- Causal Masking
- Decoder Blocks
- GPT Architecture

---

## Part V — Training

- Dataset Creation
- DataLoader
- Training Loop
- Cross Entropy
- Optimizers
- Learning Rate Scheduling
- Mixed Precision
- Gradient Checkpointing

---

## Part VI — Scaling

- CUDA
- Tensor Cores
- Distributed Data Parallel
- Fully Sharded Data Parallel
- DeepSpeed
- Pipeline Parallelism

---

## Part VII — Modern LLM Techniques

- Flash Attention
- RoPE
- GQA
- MQA
- KV Cache
- Speculative Decoding
- Mixture of Experts

---

## Part VIII — Fine-Tuning

- Instruction Tuning
- LoRA
- QLoRA
- RLHF

---

## Part IX — Deployment

- Quantization
- GGUF
- GPTQ
- AWQ
- FastAPI
- vLLM
- TensorRT-LLM
- Kubernetes

---

# 📂 Repository Structure

```
build-gpt-from-scratch/

├── book/
│   ├── chapter-01/
│   ├── chapter-02/
│   ├── chapter-03/
│   └── ...
│
├── code/
│   ├── chapter01/
│   ├── chapter02/
│   └── ...
│
├── notebooks/
│
├── datasets/
│
├── diagrams/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 📖 Chapters

| # | Chapter | Status |
|---|---------|--------|
| 1 | What is a Language Model? | 🚧 |
| 2 | Mathematics Behind Transformers | ⏳ |
| 3 | Building a Tokenizer | ⏳ |
| 4 | Embeddings | ⏳ |
| 5 | Self Attention | ⏳ |
| 6 | Multi-Head Attention | ⏳ |
| 7 | Transformer Blocks | ⏳ |
| 8 | GPT Architecture | ⏳ |
| 9 | Training a GPT | ⏳ |
| 10 | Fine-Tuning | ⏳ |
| 11 | Inference | ⏳ |
| 12 | Deployment | ⏳ |

---

# 💻 Code Philosophy

Every component is built from scratch.

Instead of writing:

```python
nn.MultiheadAttention(...)
```

We'll build:

```python
class MultiHeadAttention(nn.Module):

    def forward(self, x):

        q = ...
        k = ...
        v = ...

        scores = ...
        weights = ...
        output = ...

        return output
```

Every tensor.

Every matrix multiplication.

Every gradient.

Every line explained.

---

# 🎨 Learning Style

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

↓

Interview Questions
```

---

# 🧮 What Makes This Different?

Unlike most AI tutorials, this project explains:

- Why attention works
- Why RoPE exists
- Why RMSNorm replaced LayerNorm
- How FlashAttention works internally
- CUDA kernel execution
- GPU memory optimization
- FLOPs analysis
- Tensor dimensions at every step
- Performance bottlenecks
- Production deployment strategies

Nothing is treated as a "black box."

---

# 📊 Technologies Used

- Python 3.11+
- PyTorch
- NumPy
- Triton
- CUDA
- Hugging Face (later chapters)
- FastAPI
- Docker
- Kubernetes

---

# 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/webmakin/build-gpt-from-scratch.git

cd build-gpt-from-scratch
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the first example

```bash
python code/chapter01/bigram.py
```

---

# 📈 Progress

- [ ] Book Outline
- [ ] Chapter 1
- [ ] Chapter 2
- [ ] Tokenizer
- [ ] Transformer
- [ ] Training Pipeline
- [ ] Distributed Training
- [ ] Fine-tuning
- [ ] Inference Engine
- [ ] Deployment

---

# 🤝 Contributing

Contributions are welcome!

Whether it's:

- fixing typos
- improving explanations
- optimizing code
- adding diagrams
- benchmarking implementations
- translating chapters

Feel free to open an Issue or Pull Request.

---

# 📚 References

This project draws inspiration from the work of the broader machine learning community, including research papers and educational resources such as:

- *Attention Is All You Need* (Vaswani et al.)
- *The Illustrated Transformer* (Jay Alammar)
- Andrej Karpathy's educational GPT implementations
- PyTorch documentation
- Hugging Face documentation
- FlashAttention papers
- Llama architecture papers
- Qwen technical reports
- Mistral technical reports

All implementations in this repository are written from scratch for educational purposes.

---

# ⭐ Support

If this project helps you understand how modern language models work, consider giving it a ⭐ on GitHub.

It helps more engineers discover the project and motivates continued development.

---

# 📜 License

This project is licensed under the MIT License.

---

## 🚀 Let's Build GPT From Scratch.
