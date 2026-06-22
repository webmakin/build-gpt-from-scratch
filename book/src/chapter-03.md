# Building GPT From Scratch

## Chapter 3

# Building a Tokenizer

> *"Neural networks don't understand words — they understand integers. The tokenizer is the bridge."*

---

## 3.1 The Gap Between Text and Tensors

In Chapter 2 we built a small attention block that runs on tensors of floats.

But where do those tensors come from?

A neural network cannot read `"The cat sat on the mat"`. It can only multiply matrices.

So before any model exists, we need a function that turns **text into numbers**.

```
"The cat sat"  →  [1014, 2338, 7731]   (integer IDs)
```

And a reverse function that turns numbers back into text, so the model can reply to us.

```
[1014, 2338, 7731]  →  "The cat sat"
```

This pair of functions is called a **tokenizer**.

Building one well is one of the most underrated engineering problems in NLP. The tokenizer defines the model's vocabulary — and therefore the smallest unit of meaning it can ever learn. Choose poorly and your model wastes capacity representing common subwords as awkward sequences of single characters. Choose well and the same model trains faster and scores higher.

---

## 3.2 The Three Choices: Characters, Words, or Subwords

At first glance, splitting text into "tokens" seems easy. Just choose a granularity:

| Granularity | Vocabulary size | Example sentence |
|---|---|---|
| Character | ~256 (bytes) | `T h e   c a t   s a t` (10 tokens) |
| Word | 100k–1M+ | `The cat sat` (3 tokens) |
| Subword (BPE) | 8k–100k | `The cat sat` (3 tokens) |

**Character-level** models have tiny vocabularies but very long sequences. They spend a lot of compute learning that "q" often follows "u". GPT-2 learned English mostly from scratch this way, but at a huge compute cost.

**Word-level** models have short sequences but enormous vocabularies — and the fatal problem that any word not seen during training (every new name, typo, or piece of code) is just an unknown token. Worse, similar words like *run*, *runs*, *running*, *runner* are all unrelated indices.

**Subword** tokenizers — and Byte Pair Encoding (BPE) in particular — split rare words into common pieces. So *unhappiness* might become `["un", "happiness"]` and a brand-new word can still be encoded as a sequence of known pieces rather than `<UNK>`. This is what every modern LLM uses: GPT-2, GPT-4, Llama, Mistral, Qwen, Claude — all BPE variants.

This chapter builds a BPE tokenizer from scratch, then shows the production version used by GPT-4.

---

## 3.3 The Byte Layer — Unicode and UTF-8

Before BPE we need one more concept: **bytes**.

Computers don't store characters. They store bytes. A byte is a number from 0 to 255.

```python
b = "A".encode("utf-8")
print(b)         # b'A'   → 1 byte
print(b[0])      # 65     → the byte value
```

**Unicode** assigns a number ("code point") to every character in every language. *A* is U+0041, 🙂 is U+1F642, 你 is U+4F60.

**UTF-8** is the most common encoding that turns those code points into bytes. ASCII characters take 1 byte. Latin-1 accented characters take 2. Chinese, Arabic, and emoji take 3–4.

```python
for s in ["A", "é", "你", "🙂"]:
    b = s.encode("utf-8")
    print(f"{s!r:6} → {len(b)} bytes: {list(b)}")
# 'A'     → 1 byte:  [65]
# 'é'    → 2 bytes: [195, 169]
# '你'    → 3 bytes: [228, 189, 160]
# '🙂'  → 4 bytes: [240, 159, 153, 130]
```

Why does this matter for BPE? Because BPE can operate on **bytes** (256 symbols) instead of Unicode code points (170,000+). This is the key insight of GPT-2's tokenizer: start with the 256 byte tokens, then repeatedly merge the most common adjacent pair into a new token. *Any* string — in *any* language — can always be encoded as bytes, so you never get an `<UNK>` token. The vocabulary is built on top of those bytes.

---

## 3.4 What Is a Tokenizer?

Formally, a tokenizer is two functions:

```
encode(text: str)  -> list[int]     # text → integers
decode(ids:  list[int]) -> str      # integers → text
```

That's the entire API. Everything else — Unicode normalization, regex pre-splitting, special tokens, byte-level merges — is implementation detail to make the encoding efficient and the model's life easier.

Let's build the absolute dumbest possible tokenizer first: a character-level one.

```python
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


text = "hello world"
tok = CharTokenizer(text)

print(tok.chars)               # [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
print(tok.encode("hello"))     # [3, 1, 4, 4, 5]
print(tok.decode([3, 1, 4, 4, 5]))  # 'hello'
```

This works! Every string round-trips. The problem is that the vocabulary is the size of the text — useless for anything beyond a toy.

---

## 3.5 Byte Pair Encoding — The Algorithm

BPE was originally a data compression algorithm (Gage, 1994). In 2016, Sennrich et al. applied it to NLP tokenization. The idea is simple and beautiful:

> **Repeatedly find the most common pair of adjacent tokens and merge them into a new token.**

Start with a vocabulary of 256 byte tokens. After enough merges, you have a vocabulary of tens of thousands of tokens that are tuned to the statistics of your training corpus.

Here's the algorithm in pseudocode:

```
vocab = 256 byte tokens (0..255)

repeat N times:
    find the most frequent adjacent pair (a, b) in the corpus
    create a new token, id = len(vocab)
    replace every occurrence of (a, b) with id
    vocab += 1
```

That's it. After N merges, every original byte sequence can be re-encoded using the new tokens. The merges are stored in order, so encoding is just replaying the merges.

---

## 3.6 Building BPE From Scratch

Let's implement it on a tiny corpus and watch the vocabulary grow.

```python
from collections import Counter, defaultdict


def get_pair_counts(ids):
    """Count every adjacent pair in a sequence of token IDs."""
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with `new_id`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ── Tiny training corpus (UTF-8 bytes) ─────────────────────────

text = "the cat sat on the mat. the cat ate the rat."
ids = list(text.encode("utf-8"))   # start from raw bytes

print(f"Initial vocab: 256 bytes, sequence length {len(ids)}")

# ── Run 20 merges ──────────────────────────────────────────────

num_merges = 20
merges = {}   # (a, b) -> new_id  — replayed at encode-time

for i in range(num_merges):
    pair_counts = get_pair_counts(ids)
    if not pair_counts:
        break
    pair = pair_counts.most_common(1)[0][0]
    new_id = 256 + i
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    print(f"Merge {i+1:2}: {pair} → {new_id}  "
          f"(pair count was {pair_counts[pair]}, seq len now {len(ids)})")
```

Sample output:

```
Merge  1: (116, 104) → 256   (pair count was 8, seq len now 47)
Merge  2: (104, 101) → 257   (pair count was 7, seq len now 41)
Merge  3: (32, 116) → 258    (pair count was 6, seq len now 37)
...
```

Notice what happens: the most common pairs are exactly the common letter combinations — `th`, `he`, ` t` (space-t). These become single tokens. After 20 merges, the same text compresses to ~half its original byte count.

This is *learning a vocabulary from data*. The model that consumes these IDs gets a much shorter sequence to attend over, with tokens that already carry meaning.

---

## 3.7 Encoding and Decoding

To encode new text, we replay the merges in the order they were learned:

```python
def encode(text, merges):
    ids = list(text.encode("utf-8"))
    # merges are ordered by creation time
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids, itos):
    """Given IDs and the inverse vocabulary, return the original string."""
    out = bytearray()
    for i in ids:
        if i < 256:
            out.append(i)
        else:
            # multi-byte tokens: look up the byte sequence
            out.extend(itos[i])
    return out.decode("utf-8", errors="replace")
```

For decode to recover the original bytes, we have to remember which bytes each merged token represents. A common trick: store the merges as a list of `(pair, original_bytes)` and reconstruct `itos` from them.

```python
# Build full inverse vocab
itos = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():
    itos[new_id] = itos[a] + itos[b]

# Test round trip
test = "the cat"
encoded = encode(test, merges)
decoded = decode(encoded, itos)
print(f"{test!r} → {encoded} → {decoded!r}")
assert decoded == test
```

The round trip works. That's the entire correctness check for any tokenizer.

---

## 3.8 Real Production Tokenizers

Our toy BPE works, but production tokenizers add two important refinements:

**1. Pre-tokenization with regex.** GPT-2's tokenizer splits on whitespace and punctuation *before* running BPE. This prevents merges from crossing word boundaries (so `"dog"` and `"dog."` share most tokens but not the final period). The regex looks roughly like:

```
\s+\w|  | ?\w+ |  ?[\W]+
```

**2. Byte-level base vocabulary.** Instead of starting from Unicode code points (170k+ symbols), start from 256 raw bytes. This makes the tokenizer *total* — every possible string can be encoded, no `<UNK>` ever needed. GPT-2 maps bytes to "printable" Unicode characters (`Ġ` for space, `Ċ` for newline) for the BPE merges, then re-maps back to bytes at decode time. The result is invisible to the user.

`tiktoken` is OpenAI's fast Rust-backed implementation of this. It's already in `requirements.txt`:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
ids = enc.encode("Hello, world! 你好 🙂")
print(ids)
print(enc.decode(ids))
```

Numbers like `9906` correspond to tokens like `Hello`. To see what each ID actually represents:

```python
for i in ids:
    print(f"  {i:6} → {enc.decode([i])!r}")
#    9906 → 'Hello'
#      11 → ','
#    1917 → ' world'
#       0 → '!'
```

Notice that the leading space in `' world'` is part of the token — a common convention that lets the tokenizer distinguish ` world` (after a space) from `world` (at the start of a string). Emojis and Chinese characters usually span multiple tokens because their UTF-8 byte sequences are long and no merge has compressed them.

`tiktoken` is *fast* — about 10x faster than pure Python BPE — because the merging is implemented in Rust with a precomputed lookup table.

---

## 3.9 Special Tokens

Beyond the BPE merges, every tokenizer reserves a few IDs for control:

| Token | Purpose | Example ID (GPT-2) |
|---|---|---|
| `<\|endoftext\|>` | End of a document | 50256 |
| `<\|padding\|>` | Pads short sequences in a batch | varies |
| `<\|unk\|>` | Unknown (rarely used in BPE) | varies |

These are added to the vocabulary *after* BPE training. They are never produced by the merge process — they exist to tell the model "this is a boundary" or "ignore this position."

During training, the model learns that seeing `<|endoftext|>` means "stop generating." This is how ChatGPT knows when to stop.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

# Encode with a special end-of-text token
ids = enc.encode("Hello world", allowed_special={"<|endoftext|>"})
print(ids)

# Add a custom special token (one-time setup)
custom = enc._special_tokens  # internal map
print(custom)
```

In our from-scratch tokenizer, we'd reserve the top of the ID space (e.g. `vocab_size`, `vocab_size+1`, ...) for special tokens and skip them during the merge phase.

---

## 3.10 Vocabulary Size — The Tradeoff

Why not 50,000 tokens? Why not 500?

| Vocab size | Sequence length | Embedding table size (d=512) | Tradeoff |
|---|---|---|---|
| 256 (bytes) | long | 256 × 512 = 131k params | tiny table, very long sequences, slow attention |
| 8,192 | medium | 8k × 512 = 4.2M params | small models, fast training |
| 32,000 | short | 32k × 512 = 16.4M params | balanced (LLaMA-1) |
| 100,000 | very short | 100k × 512 = 51.2M params | large models, multilingual, code (GPT-4) |

A larger vocabulary shortens sequences (good for attention's O(T²) cost) but inflates the embedding table (a big fraction of total parameters). A smaller vocabulary does the opposite. Modern LLMs sit in the 32k–100k range. Llama 2 used 32k, GPT-4 uses ~100k for its `cl100k_base` encoding.

For our Small Language Model later in this book, we'll use roughly 4,000–8,000 tokens — enough to keep the table small but enough to merge common subwords.

---

## 3.11 The Full Pipeline

Putting it all together, a modern tokenizer does this:

```
1. Input text:    "Hello, world!"
2. Encode UTF-8:  bytes = [72, 101, 108, 108, 111, 44, ...]
3. Regex split:   chunks = ["Hello", ",", " world", "!"]
4. For each chunk, replay BPE merges → list of IDs
5. Append special tokens (e.g. <|endoftext|>)
6. Output:        [9906, 11, 1917, 0]
```

Reversing this is exactly the encode pipeline in reverse. Any tokenizer — ours, GPT-4's, Llama's — is the same idea with different vocabulary and merge lists.

---

## Chapter Summary

- **Tokenization converts text to integer IDs** so a neural network can process language.
- **BPE** repeatedly merges the most frequent adjacent token pair, learning a vocabulary from data statistics.
- **Byte-level BPE** starts from 256 raw bytes, guaranteeing every string is encodable (no `<UNK>` token).
- **Regex pre-tokenization** prevents merges from crossing word or punctuation boundaries.
- **Special tokens** mark boundaries (`<|endoftext|>`), padding, and other control signals.
- **`tiktoken`** is OpenAI's fast Rust implementation — `tiktoken.encoding_for_model("gpt-4")` gives you GPT-4's tokenizer in one line.
- The encode/decode round trip is the **correctness check** for any tokenizer.

In Chapter 4, these integer IDs become the input to an **embedding layer** — a learnable lookup table that turns each token into a dense vector a neural network can work with.

---

## Exercises

1. **Run the from-scratch BPE** in `code/chapter03/tokenizer.py` on a paragraph from Project Gutenberg. How many merges does it take before the average token length reaches 2 bytes? 3 bytes?
2. **Vocabulary size sweep.** Train BPE with `num_merges = 100, 500, 2000, 10000` on a small corpus. Plot the final sequence length for a fixed test text against `num_merges`. Where do you see diminishing returns?
3. **`tiktoken` exploration.** Use `enc.encode("...")` on (a) English, (b) Chinese, (c) Python source code, (d) base64 data. Which compresses most efficiently (fewest tokens per character)? Why?
4. **Build a chat tokenizer.** Extend `CharTokenizer` from §3.4 with a `BOS` and `EOS` token. Encode `"<BOS>hi<EOS>"` and confirm the round trip preserves them.
5. **Special tokens in `tiktoken`.** `enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})` returns a single ID. What ID is it? Decode it back. Why is `allowed_special` needed at all?
6. **Tokenization failure modes.** Find three real strings that tokenize badly (more tokens than you'd expect). Are they code, math, URLs, emojis, or something else? This is the single biggest source of "the model can't count letters" jokes online.

The full implementation is in `code/chapter03/tokenizer.py` — run it, modify it, and watch the vocabulary grow.