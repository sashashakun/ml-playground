# My ML Learning Notes

Personal notes and key insights from working through the ML curriculum.

---

## Module 1: Foundations

### Lesson 1: NumPy Basics

#### Why Matrix Multiplication is Essential in ML

**The core insight:** ML is fundamentally about transforming data. Matrix multiplication is the most efficient way to apply many transformations at once.

```python
# Single neuron (slow, sequential):
output = w1*x1 + w2*x2 + w3*x3

# Matrix form (fast, parallel):
output = X @ W  # Same math, GPU does it simultaneously
```

**Why it matters:**
- A neural network with 1000 inputs × 1000 neurons = 1,000,000 operations
- With matrices, GPUs can do millions of these **in parallel**
- This is why ML only became practical when GPUs got powerful

**Mental model:** Matrix multiplication is like batch processing for math. Instead of asking "what's 2+3?" a million times, you ask "what's this entire spreadsheet plus this other spreadsheet?" once.

---

#### The Formula: `output = input @ weights`

This represents a **linear transformation** - the core building block of neural networks.

**Intuition:** Each weight "votes" on how important each input feature is.

```python
# Example: predicting house price
input = [sqft, bedrooms, age]       # Your data
weights = [100, 5000, -200]         # Learned importance of each feature

price = input @ weights
# = sqft*100 + bedrooms*5000 + age*(-200)
```

**In neural networks:**
- `input` = your data (words as numbers, pixels, etc.)
- `weights` = **learned parameters** (the network discovers these during training)
- `output` = transformed data, one step closer to prediction

Each layer does: `output = input @ weights + bias` → then activation function

---

#### `*` vs `@` - The Critical Difference

| Operator | Name | What it does |
|----------|------|--------------|
| `*` | Element-wise | Multiply matching positions |
| `@` | Matrix multiplication | Dot product / linear algebra |

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a * b  # → [1*4, 2*5, 3*6] = [4, 10, 18]  (element-wise)
a @ b  # → 1*4 + 2*5 + 3*6 = 32           (dot product)
```

**2D Matrix Example:**
```python
A = [[1, 2],      B = [[5, 6],
     [3, 4]]           [7, 8]]

# Element-wise (*): multiply corresponding cells
A * B = [[5, 12],
         [21, 32]]

# Matrix multiplication (@): rows of A dot columns of B
A @ B = [[19, 22],
         [43, 50]]
```

**When to use which:**
- `*` → Scaling, masking, per-element operations
- `@` → Layer transformations, computing weighted sums

---

#### GPU Efficiency

Both `*` and `@` CAN run on GPU. But `@` benefits MORE because:

| Aspect | `*` | `@` |
|--------|-----|-----|
| GPU efficiency | Good | **Excellent** |
| Why | Independent ops | Data reuse + regularity |

**Why GPUs love `@`:** Matrix multiplication has a "multiply-accumulate" pattern that GPUs are specifically designed for. They can load weights once and broadcast across thousands of cores.

---

#### Restaurant Mental Model

- **Element-wise (`*`):** "Double the price of every menu item" - each item × its own multiplier
- **Matrix mult (`@`):** "Calculate the total bill" - sum of (quantity × price) for all items

---

#### Words as Vectors: The Core Concept

**Your math knowledge:** A vector is a line from one point to another (direction + magnitude).

**ML meaning:** Same thing! A list of numbers = coordinates of a point = vector from origin to that point.

```python
# 2D: easy to visualize
vector = [3, 2]  # → point at (3, 2) → vector from origin to there

# 5D: can't visualize, but same math
vector = [0, 1, 0, 0, 0]  # → point in 5-dimensional space
```

**Key insight:** In ML, "vector" means "list of numbers" - but we call it vector because all geometric intuitions (distance, direction, similarity) still apply!

---

#### One-Hot Encoding: Words as Vectors

Each word gets its own axis (dimension):

```python
vocabulary = ["I", "love", "cats", "dogs", "hate"]  # 5 words = 5 dimensions

"I"    → [1, 0, 0, 0, 0]  # Vector from origin to (1,0,0,0,0)
"love" → [0, 1, 0, 0, 0]  # Vector from origin to (0,1,0,0,0)
"cats" → [0, 0, 1, 0, 0]  # etc.
```

**Sentence as matrix:** Stack word vectors together

```
"I love cats" =  [[1, 0, 0, 0, 0],   ← "I" vector
                  [0, 1, 0, 0, 0],   ← "love" vector
                  [0, 0, 1, 0, 0]]   ← "cats" vector
```

Matrix is just storage - each row is still a vector starting from origin!

---

#### Display Dimensions vs Mathematical Dimensions

**Common confusion:** The matrix LOOKS 2D on screen, but that's just display!

| Term | Meaning | Example |
|------|---------|---------|
| Display dimensions | Rows × Columns on screen | 3×5 table |
| Mathematical dimensions | Numbers in ONE vector | 5 numbers = 5D |

```python
"love" → [0, 1, 0, 0, 0]
          ↑  ↑  ↑  ↑  ↑
          5 numbers = 5 dimensions (even though written as one line)
```

**Analogy:** Your address `[Street: 15, Building: 7, Floor: 3, Apt: 42]` is written on one line but describes a point in 4D space!

---

#### Why Dimensions = Vocabulary Size

```python
5 words   → 5 dimensions   → vectors of length 5
100 words → 100 dimensions → vectors of length 100
50,000 words (real English) → 50,000 dimensions!
```

Each word needs its own axis to have a unique position.

---

#### One-Hot Encoding Limitation: Everything Equally Far!

```
    "dog" axis
         ↑
       1 │  • "dog" (0,1)
         │ ↖
         │   ↖  distance = √2
         │     ↖
    ─────┼───────•────→ "cat" axis
         0       1
                "cat" (1,0)
```

**Problem:**
- Every word is distance 1 from origin
- Every word is distance √2 from every other word
- "cat" ↔ "dog" (both pets) = same distance as "cat" ↔ "quantum_physics"!

**No meaning captured!** This is why embeddings (Module 5) were invented.

---

#### Where Did Vectors Come From? (History)

**Not invented for ML!** Centuries old math:

| Era | Development |
|-----|-------------|
| 1600s | Descartes invented coordinates: any point = list of numbers |
| 1800s | Linear algebra: basis vectors, n-dimensional spaces |
| 1950s-60s | CS asked: how to represent categories as numbers? |
| 1980s+ | ML adopted one-hot encoding as standard |

**The solution for categories:**
```python
# Can't use: red=1, green=2, blue=3 (implies green is "between"!)
# Solution: each category gets own axis
red   = [1, 0, 0]
green = [0, 1, 0]  # All independent, none "between" others
blue  = [0, 0, 1]
```

**ML's real innovation:** Not vectors themselves, but *learning* better vectors (embeddings) where similar things ARE close together.

---

#### Mental Models

**One-hot encoding:** Every person gets their own planet. Easy to find, but no concept of neighbors.

**Embeddings (coming in Module 5):** Everyone lives in a city. Similar people in same neighborhoods. You can walk from "love" to "like" but "hate" is across town.

---

*Notes from: Lesson 1 Q&A session*

---

### Lesson 2: Data Intuition

#### Bag-of-Words (BoW): Text → Numbers

**What it is:** Convert text to numbers by counting words. "Bag" because we throw words into a bag and **lose their order**.

```python
Sentence: "I love cats and I love dogs"

Vocabulary: ["I", "love", "cats", "and", "dogs"]

Bag-of-Words vector: [2, 2, 1, 1, 1]
                      ↑  ↑  ↑  ↑  ↑
                      I  love cats and dogs
                     (2) (2) (1) (1) (1)
```

---

#### Why It's a "Vector"

A vector is a list of numbers = a point in space. Each sentence becomes a **point in vocabulary-dimensional space**:

```python
"I love cats" → [1, 1, 1, 0, 0]  # Point in 5D space
"I love dogs" → [1, 1, 0, 0, 1]  # Different point in 5D space
```

---

#### The "Bag" Problem: Word Order Lost

These sentences have **identical** BoW vectors:

```
"The movie was not good"  →  {the:1, movie:1, was:1, not:1, good:1}
"The movie was good not"  →  {the:1, movie:1, was:1, not:1, good:1}  # Same!
```

Even worse - opposite meanings, similar vectors:
```
"I love this, I don't hate it"  vs  "I hate this, I don't love it"
```

---

#### Connection to One-Hot Encoding

| Method | Vector for "love" | What it captures |
|--------|-------------------|------------------|
| One-hot | `[0,1,0,0,0]` | Which word (binary) |
| Bag-of-Words | `[2,2,1,1,1]` | Word counts in sentence |

**BoW = summing one-hot vectors** for every word in the sentence!

---

#### Why BoW Still Matters

**Dumb but useful:** Despite losing word order, BoW works well for:
- Spam detection
- Topic classification
- Simple sentiment analysis

**For order-sensitive tasks, you need:**
- Sequence models (Module 6) - RNNs remember order
- Transformers (Module 8) - Attention captures relationships

---

*Notes from: Lesson 2 Q&A session*
