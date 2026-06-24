# Building GPT From Scratch

## Chapter 1

# What Is a Language Model?

> *"Before we build intelligence, we first need to understand what it means for a machine to learn language."*

---

## 1.1 Introduction

If you ask someone today,

> *"What is ChatGPT?"*

Most people will answer:

> "It's an AI."

Ask an engineer:

> "It's a Large Language Model."

Ask a researcher:

> "It's a Transformer."

While none of these answers are wrong, they don't explain **how** it actually works.

At its core, ChatGPT—and every modern language model—is solving an astonishingly simple problem:

> **Given everything it has seen so far, what comes next?**

That's it.

No built-in understanding of grammar.

No knowledge of the world at initialization.

No predefined rules about English, French, or Python.

Only one objective:

> **Predict the next token as accurately as possible.**

Yet from this deceptively simple objective emerge reasoning, coding, translation, summarization, and even creative writing.

How?

That is the question this book answers.

---

# 1.2 A Child Learning Language

Imagine a three-year-old child.

They hear sentences every day:

> I want milk.

> The dog is barking.

> Dad is coming home.

Initially, these sounds have no meaning.

But after hearing millions of examples, the child begins noticing patterns.

When someone says:

> I want...

The next word is often:

* water
* food
* milk
* candy

Eventually the child starts predicting the next word before it is spoken.

Without realizing it, the child is learning a probability distribution over language.

Modern language models do something remarkably similar.

The difference is scale.

A child may hear tens of millions of words during early development.

A modern SLM might train on tens of billions of tokens.

---

# 1.3 Prediction Is Intelligence

This sounds almost unbelievable.

Can prediction alone create intelligence?

Let's consider a simple game.

Suppose I repeatedly show you these sequences:

```
2, 4, 6, 8, ?
```

You immediately answer:

```
10
```

Why?

Because your brain has learned patterns.

Now another:

```
Monday
Tuesday
Wednesday
?
```

You answer:

```
Thursday
```

Again, prediction.

Now consider English.

```
The sky is
```

Most people predict:

```
blue
```

Again,

prediction.

Language itself is fundamentally a prediction problem.

---

# 1.4 What Is a Language Model?

A language model is simply a mathematical function.

We can write it as:

\\[
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
\\]

This notation looks intimidating, but it means only one thing:

> **What is the probability of the next token given every previous token?**

Suppose we have the sentence:

```
I love eating
```

A language model might assign probabilities like this:

| Next Token      | Probability |
| --------------- | ----------: |
| pizza           |        0.42 |
| pasta           |        0.23 |
| sushi           |        0.11 |
| burgers         |        0.05 |
| apples          |        0.04 |
| everything else |        0.15 |

Notice something important.

The model does **not** know the correct answer.

Instead, it estimates a probability distribution over all possible next tokens.

The output is uncertainty, not certainty.

---

# 1.5 Deterministic Programs vs Learned Models

Traditional software is written as explicit rules.

```python
if age >= 18:
    can_vote = True
else:
    can_vote = False
```

The programmer defines every decision.

Language models work differently.

There is no rule like:

```python
if previous_word == "love":
    next_word = "pizza"
```

Instead, the model learns patterns from data.

Training changes millions—or even billions—of parameters until the model can make increasingly accurate predictions.

---

# 1.6 The Simplest Language Model

Before neural networks, researchers built statistical language models.

Imagine a tiny dataset:

```
I like pizza
I like pasta
I like coffee
You like pizza
```

We can count what follows the word **like**.

| Word   | Count |
| ------ | ----: |
| pizza  |     2 |
| pasta  |     1 |
| coffee |     1 |

The probability of each next word is simply its frequency divided by the total count:

\\[
P(\text{pizza} \mid \text{like}) = \frac{2}{4} = 0.5
\\]

\\[
P(\text{pasta} \mid \text{like}) = \frac{1}{4} = 0.25
\\]

\\[
P(\text{coffee} \mid \text{like}) = \frac{1}{4} = 0.25
\\]

We've just built a basic statistical language model—no neural networks required.

Its limitation is obvious: it only considers the immediately preceding word. If the context changes, its predictions do not.

---

# 1.7 Building Our First Language Model

Let's implement a tiny bigram language model in Python.

```python
from collections import defaultdict

text = """
I like pizza
I like pasta
I like coffee
You like pizza
"""

words = text.lower().split()

bigrams = defaultdict(lambda: defaultdict(int))

for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    bigrams[current_word][next_word] += 1

print(dict(bigrams["like"]))
```

Output:

```python
{
    'pizza': 2,
    'pasta': 1,
    'coffee': 1
}
```

At this point, we are simply counting occurrences. The next step is to convert these counts into probabilities.

```python
def predict_next(word):
    options = bigrams[word]
    total = sum(options.values())

    probabilities = {}

    for token, count in options.items():
        probabilities[token] = count / total

    return probabilities

print(predict_next("like"))
```

Output:

```python
{
    'pizza': 0.5,
    'pasta': 0.25,
    'coffee': 0.25
}
```

Congratulations! You have built your first language model. It is small, limited, and far from GPT, but the core idea is the same: estimate the probability of the next token given some context.

---

# 1.8 Why This Model Fails

Now test it with:

```
I really like
```

Our model still predicts:

```
pizza
```

It cannot use the word "really" because it only remembers one previous token. This limitation motivates more advanced models.

What if we remembered two words? Or ten? Or every previous word in the sentence?

These questions led to n-gram models, recurrent neural networks, and eventually the Transformer architecture.

---

# Chapter Summary

In this chapter, we learned that:

* A language model predicts the next token based on previous tokens.
* Early language models relied on simple frequency counts.
* Prediction can be expressed as a probability distribution rather than a single answer.
* Neural language models replace hand-written rules with learned parameters.
* Even the simplest bigram model demonstrates the central idea behind modern language modeling.

---

# Exercises

1. Modify the bigram model to ignore punctuation and compare the predictions.
2. Extend the model to use two previous words (a trigram model). How does its accuracy change?
3. Add a function that samples the next word randomly according to the predicted probabilities instead of always selecting the most frequent one.
4. Measure how the model behaves when trained on a much larger text corpus, such as a book from Project Gutenberg.

---

This chapter lays the conceptual foundation for everything that follows. In **Chapter 2: Mathematics of Language Models**, we'll begin turning these simple ideas into the tensor operations that power modern SLMs. We'll introduce vectors, matrices, tensors, and show how text becomes numbers that a neural network can process. From there, every subsequent chapter will build directly toward implementing a GPT-style model from scratch.
