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

---

### Lesson 3: Linear Regression

#### The Line IS the Model IS the Prediction

**Key insight:** In linear regression, the model is literally a line. Every point on that line is a prediction.

```
The line = the model = all possible predictions at once
```

When you "train" a model, you're finding the **best line** (slope + intercept). Once found, predicting is just reading a value off the line.

```python
# The model learns just TWO numbers:
price = slope * sqft + intercept
#       ↑               ↑
#       steepness        where line crosses y-axis
```

---

#### Predictions vs Real Data — Don't Confuse Them!

```
Price
  |        x          x = real sold houses (training data)
  |    x  /
  |      / x
  |    /x        ● = prediction (always ON the line)
  |   ●
  |  /  x
  | / x
  |/_______________
       Square feet
```

| What | Where it lives | Why |
|------|---------------|-----|
| **Prediction** | Exactly ON the line | The line IS the prediction |
| **Real data** | Scattered AROUND the line | Reality is messy |
| **Error** | Gap between real and line | What we try to minimize |

---

#### Why Real Data Doesn't Fall on the Line

The model might only know square footage, but real prices depend on many things:
- **Missing features:** location, bathrooms, year built, garage
- **Human randomness:** desperate sellers, bidding wars, market shifts
- **Measurement noise:** rounding, different measurement methods

Two identical-size houses can sell for very different prices because of factors the model can't see.

---

#### Loss Function: "How Wrong Is My Line?"

The loss function measures how far the real data points are from the line:

```python
Loss = average of (actual_price - predicted_price)²
```

Training = moving the line around until total squared distance is **as small as possible**.

**Why squared?** Squaring makes big errors count way more than small ones, so the model avoids being very wrong about any single house.

---

#### The Formula: `y = mx + b`

**Origin:** 1600s algebra (Descartes). Any straight line can be described with just two numbers.

| Symbol | Name | Meaning |
|--------|------|---------|
| `m` | Slope | How much y changes when x goes up by 1 |
| `b` | Intercept | Where the line crosses the y-axis (value when x = 0) |
| `x` | Input | The feature you're using to predict |
| `y` | Output | The prediction |

```
y = mx + b
↑   ↑    ↑
│   │    └── where you start (when x = 0)
│   └─────── how fast you climb per step
└─────────── the result
```

**Housing example:**
```python
price = 200 * sqft + 50000
#       m=200        b=50,000
#       ($200/sqft)  (base price: land, permits, etc.)

# 1000 sqft → $250,000
# 1500 sqft → $350,000
# 2000 sqft → $450,000
```

**ML vocabulary:** In ML, `m` is called the **weight** (`w`) — same thing, different letter. "Weight" means "how important is this input."

```python
# Algebra:  y = mx + b
# ML:       y = wx + b       (one feature)
# ML:       y = w1*x1 + w2*x2 + w3*x3 + b   (multiple features)
# Matrix:   y = X @ W + b    (same thing, connects back to Lesson 1!)
```

**Training = finding the best `m` and `b`.** That's it — the algorithm searches for two numbers that produce the lowest loss.

---

#### Irreducible Error

Even the perfect line can't eliminate all error. The leftover gap is called **irreducible error** — noise from information the model simply doesn't have.

If all points DID fall perfectly on a line, you wouldn't need ML — a simple formula would do!

---

#### Gradient Descent: Finding the Best w and b

**What it is NOT:** trying random w and b values and picking the best (brute force — too slow).

**What it IS:** using the slope (gradient) of the loss curve to know **which direction** to step, then stepping downhill.

```
Аналогия: ты с завязанными глазами на холме.
Высота = ошибка (чем ниже, тем лучше).
Щупаешь ногой — куда наклон вниз? Шагаешь туда. Повторяешь.
```

**The algorithm:**
```
1. Start with random w and b
2. Calculate MSE (how wrong am I?)
3. Calculate gradient (which direction is downhill?)
4. Update: w = w - learning_rate × ∂L/∂w
           b = b - learning_rate × ∂L/∂b
5. Repeat until gradient ≈ 0 (reached the bottom)
```

---

#### Derivative (Производная) = Slope of a Curve

For a straight line, slope is the same everywhere — just `m`.
For a curve, slope changes at every point — that's the **derivative**.

**How to calculate:** take two very close points and use the regular slope formula:

```
y = x²

Point 1: x = 3       → y = 9
Point 2: x = 3.001   → y = 9.006

derivative ≈ (9.006 - 9) / (3.001 - 3) = 6

Same as the calculus formula: derivative of x² = 2x → 2×3 = 6 ✓
```

The derivative is NOT a new concept — it's the same rise/run formula from school, applied to a curve with a tiny step.

---

#### Partial Derivatives (Частные производные) — One Direction at a Time

With two parameters (w and b), the loss is a 3D surface (a "bowl"):

```
     Loss (up)
        |
        |   ╱‾‾╲
        |  ╱ 🔴 ╲    ← bottom of the bowl = best w and b
        | ╱______╲
       w              b
```

A **partial derivative** = slope in ONE direction, other variables frozen:

| Derivative | Meaning | Analogy |
|-----------|---------|---------|
| `∂L/∂w` | How loss changes when only w moves (b fixed) | Look east only |
| `∂L/∂b` | How loss changes when only b moves (w fixed) | Look north only |

**Gradient** = vector of all partial derivatives = `[∂L/∂w, ∂L/∂b]`
Points in the direction of **steepest ascent**. Go opposite → descend.

---

#### Two Different Graphs — Don't Confuse Them!

| Graph | Axes | Purpose |
|-------|------|---------|
| Data plot (cell-3) | sqft → price | Shows data points and the prediction line |
| Loss landscape (cell-10) | weight → MSE | Shows where gradient descent "walks" |

The loss landscape in lesson 3 is 2D because bias is fixed (`b=50`). The real loss landscape with both w and b would be a 3D bowl — but 2D slice is easier to understand.

---

#### Learning Rate: Step Size

**Not random, not learned — chosen by YOU before training.**

Two types of numbers in ML:

| | Parameters (w, b) | Hyperparameters (learning rate) |
|---|---|---|
| Who chooses? | Model learns them | You set them |
| Starting value | Random | You pick a specific number |
| Changes during training? | Yes, every step | No, stays fixed* |

Learning rate controls **how far** to step (gradient says **which direction**):

```python
w = w - learning_rate × gradient

gradient = -200 (go right)
lr = 0.001  →  step = 0.2    (small step)
lr = 0.01   →  step = 2.0    (normal step)
lr = 0.1    →  step = 20.0   (too far — overshoot!)
```

**How to choose in practice:**
```
Start with 0.001 (default in Adam optimizer, works 80% of the time)
Loss not dropping?        → try 0.01  (bigger steps)
Loss exploding/jumping?   → try 0.0001 (smaller steps)
```

| Learning rate | Effect |
|--------------|--------|
| Too small (0.0001) | Works but painfully slow |
| Just right (0.001) | Smooth convergence |
| Too large (0.1) | Overshoots minimum, diverges! |

There is no magic formula — it's trial and error. That's why it's called a **hyper**parameter (above the model, tuned by human).

---

#### Sentiment Score: Applying Linear Regression to Text

**Sentiment** = настроение/тональность текста. **Score** = число от 0 до 1.

```
0.0 = very negative    "Terrible movie, waste of time"
0.5 = neutral          "I watched a movie yesterday"
1.0 = very positive    "Best movie of my life!"
```

**How it works in lesson 3 (cell-19):** count positive words minus negative words, then linear regression:

```python
sentiment_score = w × (positive_words - negative_words) + b

"I love this amazing film"  → +2 → score ≈ 0.7 (positive)
"terrible boring waste"     → -3 → score ≈ 0.2 (negative)
"I watched a film"          →  0 → score ≈ 0.5 (neutral)
```

Same `y = wx + b` formula — just applied to text instead of house prices.

**Connection to Lesson 2 (BoW):**
```
BoW:               [2, 1, 0, 1, 3]   (count of each word)
Sentiment feature:  +2                (pos - neg, one number)
```
Sentiment feature is a simplified BoW — one number for overall mood instead of a count per word.

**Real-world uses:** product reviews, social media monitoring, support ticket prioritization, financial news analysis.

**Limitation:** breaks on sarcasm — "Oh *great* service" has a positive word but negative meaning. Need Transformers (Module 8) for that.

---

*Notes from: Lesson 3 Q&A session*

---

## Module 2: Classification

### Lesson 4: Logistic Regression

#### Why Linear Regression Fails for Classification

Linear regression outputs a straight line that goes from -∞ to +∞. Three problems:

| Problem | Linear regression | Logistic (sigmoid) |
|---------|------------------|-------------------|
| Output range | Can be -5 or +3 (not valid probabilities) | Always (0, 1) |
| Sensitivity to outliers | One extreme point shifts the entire line | Extremes get squished |
| Interpret as probability | No — 0.8 is just a point on a line | Yes — 0.8 means "80% confident" |

**Can't you just round?** You could, but outliers will shift the line and break the decision boundary for everyone else. Sigmoid doesn't have this problem.

---

#### Sigmoid: Squishing Any Number to (0, 1)

```
sigmoid(z) = 1 / (1 + e^(-z))
```

`e` = 2.71828 (Euler's number, like π). `e^(-z)` shrinks to zero for large z.

How it works — just plug in:

```
z = 0:    1 / (1 + e^0)    = 1 / (1 + 1)     = 0.5      ← middle
z = 10:   1 / (1 + e^-10)  = 1 / (1 + 0.00005) ≈ 0.99999 ← almost 1
z = -10:  1 / (1 + e^10)   = 1 / (1 + 22026)   ≈ 0.00005 ← almost 0
```

The trick: big positive z → denominator ≈ 1 → output ≈ 1. Big negative z → denominator huge → output ≈ 0.

```
input:  -∞ ──────────── 0 ──────────── +∞
output:  0 ──────────  0.5  ──────────   1

                  __________
                /
              /      ← S-shaped curve
           __/
```

---

#### Where Sigmoid Came From (Not ML!)

**1830s** — Belgian mathematician Verhulst studied population growth. Population can't grow forever (limited food/territory). He needed a formula that starts fast, then levels off → the **logistic curve** = sigmoid.

**1940-50s** — Statisticians adopted it for probability modeling (yes/no tasks) → logistic regression was born.

**1980s+** — ML used sigmoid as activation function in neural networks.

**Why this formula and not something else?**
- Output (0, 1) → reads as probability
- Easy derivative: `sigmoid(z) × (1 - sigmoid(z))` → good for gradient descent
- Mathematically connected to probability theory (log-odds)

---

#### Sigmoid in the Real World (Outside ML)

Sigmoid is a **math function borrowed by ML**, not invented for it. The S-curve appears everywhere in nature:

| Domain | Example | Why it's sigmoid |
|--------|---------|-----------------|
| **Biology** | Bacteria growth in a petri dish | Slow start → boom → plateau (food runs out) |
| **Pharmacology** | Dose-response curves | Small dose = no effect, medium = sharp response, large = plateau |
| **Economics** | Technology adoption (smartphones, internet) | Early adopters → mass growth → market saturation |
| **Physics** | Magnetization vs temperature | Smooth phase transition = S-curve |
| **Neurobiology** | Neuron firing rate vs stimulus | Weak input = silence, strong input = max firing, in between = S-curve |

**The neurobiology connection is why ML adopted sigmoid** — early neural network researchers (1940-60s) modeled biological neurons and needed a function that mimicked "fire or don't fire" with a smooth transition.

---

#### Alternatives to Sigmoid

Since sigmoid is just a convenient choice, other functions exist:

| Function | Range | Used for |
|----------|-------|----------|
| **Sigmoid** | (0, 1) | Binary classification, LSTM gates |
| **Tanh** | (-1, 1) | Same S-shape but centered at zero |
| **ReLU** | [0, ∞) | Modern neural nets — simpler, faster |
| **Softmax** | (0, 1), sums to 1 | Multi-class (cat/dog/bird) |

**Why sigmoid lost popularity in deep networks:** in deep nets, sigmoid causes "vanishing gradients" — the gradient gets tiny in the flat tails, so layers far from the output barely learn. ReLU doesn't have this problem. But sigmoid is still used as the **output layer** for binary classification and inside **LSTM/GRU gates**.

---

#### Logistic Regression = Linear + Sigmoid

```python
# Linear regression:
prediction = w * x + b                    # output: -∞ to +∞

# Logistic regression:
prediction = sigmoid(w * x + b)           # output: 0 to 1
```

Same `y = wx + b` inside, just wrapped in sigmoid. That's the only difference!

**Note:** "Logistic *regression*" is a confusing name — it's actually a **classification** method.

---

#### BCE (Binary Cross-Entropy): Why Not MSE for Classification?

**The problem with MSE + sigmoid:** In the flat "tails" of the sigmoid curve (near 0 and near 1), the gradient is nearly zero. MSE in those zones also gives a tiny gradient. Result: the model **gets stuck and stops learning**, even when its prediction is completely wrong.

**BCE solves this** by using the logarithm, which explodes toward infinity as the prediction approaches the wrong answer:

```
BCE = -mean( y·log(ŷ) + (1-y)·log(1-ŷ) )
```

---

#### How BCE Punishes Mistakes

| Situation | MSE penalty | BCE penalty |
|-----------|-------------|-------------|
| True = 1, predicted 0.9 | small | small |
| True = 1, predicted 0.5 | medium | medium |
| True = 1, predicted 0.01 | penalty ≈ 1 | penalty → **∞** ! |

**Key insight:** BCE exponentially punishes confident wrong answers.

Two graphs in notebook cell-9 show this:
- **Left graph** (true label = 1): loss curve drops as prediction → 1, explodes as prediction → 0
- **Right graph** (true label = 0): mirror image — loss explodes as prediction → 1

**Doctor analogy:** A doctor who says "99% healthy" when the patient is sick deserves a much bigger penalty than one who says "55% healthy". MSE doesn't see this difference sharply. BCE does.

---

#### Where BCE Comes From

Not invented for ML — comes from **information theory** (Shannon, 1948). "Cross-entropy" measures how different two probability distributions are. In our case: how far the predicted probabilities are from the real labels (0 or 1).

BCE + sigmoid are a mathematically natural pair: sigmoid outputs a probability, BCE measures the quality of that probability. This is why they're always used together.

---

#### How Training Works: The Model Stores Parameters, Not Data

**Common misconception:** "the model adjusts probability for each point individually."

**Reality:** The model only stores **weights and bias** (e.g., 3 numbers for 2-feature logistic regression). Probabilities are recalculated from scratch every iteration using the formula `sigmoid(X @ w + b)`.

```
Iteration 1: weights = [0, 0], bias = 0
  Point [3.0, -1.5]:  z = 3.0×0 + (-1.5)×0 + 0 = 0  → sigmoid(0)     = 0.5
  Point [-2.0, 1.0]:  z = (-2.0)×0 + 1.0×0 + 0 = 0   → sigmoid(0)     = 0.5
  (all points get 0.5 — model knows nothing yet)

Iteration 2: weights = [0.12, 0.09], bias = 0.01   ← only 3 numbers changed
  Point [3.0, -1.5]:  z = 3.0×0.12 + (-1.5)×0.09 + 0.01 = 0.235  → sigmoid = 0.56
  Point [-2.0, 1.0]:  z = (-2.0)×0.12 + 1.0×0.09 + 0.01 = -0.14  → sigmoid = 0.47
  (different probabilities — because each point has different X values)
```

**Analogy:** The model doesn't move each point — it **rotates and shifts one line** (the decision boundary). One rotation changes the distance to every point differently.

---

#### Parameters: From 3 Numbers to Trillions

The same `y = X @ W + b` scales from linear regression to GPT:

```
Linear regression (1 feature):     y = w·x + b                    → 2 params
Logistic regression (2 features):  y = sigmoid(w1·x1 + w2·x2 + b) → 3 params
Logistic regression (100 features): y = sigmoid(X @ w + b)         → 101 params
```

A neural network is **many layers of logistic regression** stacked together. Each neuron in a layer is one logistic regression, and every neuron connects to every neuron in the next layer:

```
Input (784) → Layer 1 (256) → Layer 2 (128) → Output (10)

  784 × 256 + 256 bias = 200,960
  256 × 128 + 128 bias =  32,896
  128 × 10  + 10 bias  =   1,290
                          ────────
                          235,146 parameters (simple digit recognition)
```

Real-world scale:

| Model | Parameters | Task |
|-------|-----------|------|
| Lesson 4 logistic regression | **3** | pass/fail from 2 features |
| Simple neural net | **~235K** | handwritten digit recognition |
| ResNet-50 (2015) | **~25M** | image classification |
| GPT-2 (2019) | **~1.5B** | text generation |
| GPT-4 (2023) | **~1.8T** (estimate) | everything |

The training loop is **identical** at every scale: forward pass → loss → gradients → update. Just more parameters and gradients computed through chains of layers (**backpropagation** — coming in Module 4).

This is why Lesson 1 started with matrix multiplication and GPUs — with trillions of parameters, `X @ W + b` must run for every layer, every batch, thousands of times. Without GPUs it's physically impossible.

---

*Notes from: Lesson 4 Q&A session*

---

### Lesson 5: Sentiment Analysis (Basic)

#### Words = Axes, Reviews = Points

Easy to confuse "word vector" and "review vector." Both exist, but they play different roles:

| What | Role | Example |
|------|------|---------|
| **Word** | axis (dimension) of the space | "amazing" axis, "awful" axis |
| **Review** | point in that space | `[1, 0, 0, 1, 1]` |
| **Model** | boundary dividing points into + and − | `sigmoid(X @ w + b)` |

**In Lesson 1 (one-hot):** each word is a vector (one point per word).
**In Lesson 5 (BoW):** each review is a vector (one point per review). Words become the axes.

---

#### BoW Vector = Sum of One-Hot Vectors

A review's BoW vector is literally the sum of its words' one-hot vectors:

```
Vocabulary: [amazing, awful, boring, great, movie]

"amazing great movie":
  amazing → [1, 0, 0, 0, 0]
  great   → [0, 0, 0, 1, 0]
  movie   → [0, 0, 0, 0, 1]
            ─────────────────
  sum     → [1, 0, 0, 1, 1]   ← BoW vector of the review
```

Individual words dissolve into one summary vector. The model sees counts, not word order.

---

#### The Whole Pipeline in Lesson 5

```
"amazing great movie"
        │
    preprocess (lowercase, remove punctuation, split)
        │
    BoW vectorizer → [1, 0, 0, 1, 1]   (point in 45D word-space)
        │
    sigmoid(X @ w + b) → 0.87           (probability: positive)
        │
    threshold ≥ 0.5 → "Positive"
```

Same logistic regression as Lesson 4, just applied to text. The 45 learned weights tell you which words push toward positive (high weight) and which push toward negative (low weight) — visible in cell-16/17 bar charts.

---

#### BoW Limitation: Word Relationships Lost

In BoW space, "amazing" and "fantastic" are just as far apart as "amazing" and "terrible" — each word has its own independent axis. Embeddings (Module 5) will fix this by putting words into a shared space where similar words are close together.

---

#### A Model IS Its Weights

After training, everything the model "knows" is stored in **just the weights and bias**:

```python
model.weights = [0.42, -0.87, -0.65, 0.31, ...]   # 45 numbers (one per word)
model.bias = 0.03                                    # 1 number
# Total: 46 numbers. This IS the model. Training data no longer needed.
```

Prediction is just plugging new data into the formula with these saved weights:

```python
z = new_review_vector @ model.weights + model.bias   # arithmetic
prob = sigmoid(z)                                      # probability
# No "learning" happens here — just a calculation with fixed numbers.
```

**This is true for ALL models:**

| Model | What's saved | Size |
|-------|-------------|------|
| Lesson 5 logistic regression | weights + bias | 46 numbers |
| Simple neural net | weights of all layers | ~235K numbers |
| GPT-4 | weights of all layers | ~1.8T numbers |

When you download a "model file" (`.pt`, `.h5`) — it's literally an array of numbers. "The model weighs 7GB" = 7GB of weight values.

**Two separate things:**
- **Architecture** = the formula/code (how many layers, which activation functions) — doesn't change
- **Weights** = the numbers found by training — this is what makes a model smart or dumb

Same architecture + different weights = different model. GPT-3.5 and GPT-4 use similar architecture (Transformer) but different weights → different behavior.

---

#### Cross-Validation: Don't Trust One Exam

**Problem:** One train/test split can be lucky or unlucky — especially with small datasets (like 24 reviews in Lesson 5).

**Solution:** Split the data k different ways, train and test each time, take the average:

```
Data: [A] [B] [C] [D] [E]     (5 parts)

Fold 1:  TEST [B] [C] [D] [E]   → accuracy = 85%
Fold 2:  [A] TEST [C] [D] [E]   → accuracy = 90%
Fold 3:  [A] [B] TEST [D] [E]   → accuracy = 80%
Fold 4:  [A] [B] [C] TEST [E]   → accuracy = 95%
Fold 5:  [A] [B] [C] [D] TEST   → accuracy = 85%
                                   ─────────────
                           Average: 87% ± 5%
```

Every part is test exactly once = **k-fold cross-validation** (here k=5).

**Analogy:** One train/test split = one exam (might get lucky with the questions). Cross-validation = 5 exams with different questions — average grade is much more reliable.

**When to use:**
- Small datasets (few hundred examples) → almost always
- Comparing models or hyperparameters → honest comparison
- Large datasets (millions) → one split is usually enough (law of large numbers)
- Huge models (GPT) → too expensive (5× training cost)

In scikit-learn: `cross_val_score(model, X, y, cv=5)` — one line.

---

#### fit() vs transform() vs fit_transform() — The Sklearn Pattern

Same `fit`/`predict` idea from models, but for **data preprocessing**:

| Step | What it does | Example (BagOfWords) |
|------|-------------|---------------------|
| `fit()` | Learn statistics from data | Build vocabulary (45 words) |
| `transform()` | Apply learned statistics | Convert texts → BoW vectors |
| `fit_transform()` | Both at once (shortcut) | Build vocab + convert in one call |

```python
# These two are identical:
vectorizer.fit(train_texts)
X_train = vectorizer.transform(train_texts)

X_train = vectorizer.fit_transform(train_texts)   # same thing, one line
```

**Critical rule: fit only on train data!**

```python
# ✅ CORRECT:
X_train = vectorizer.fit_transform(train_texts)   # fit + transform on train
X_test = vectorizer.transform(test_texts)          # only transform on test

# ❌ WRONG:
X_test = vectorizer.fit_transform(test_texts)      # re-fits on test = data leakage!
```

Re-fitting on test = the model "peeks" at test data. Vocabulary/statistics must come only from training data.

**Same pattern everywhere in sklearn:**

| Purpose | Learn | Apply |
|---------|-------|-------|
| **Model** | `model.fit(X, y)` | `model.predict(X_new)` |
| **Preprocessing** | `scaler.fit(X)` | `scaler.transform(X_new)` |

Models learn **weights**, preprocessors learn **statistics** (vocabulary, mean, scale). Principle is the same: learn on train, apply to anything.

---

*Notes from: Lesson 5 Q&A session*

---

## Module 3: Neural Networks

### Lesson 7: The Perceptron

#### Same Formula, Different Names

The perceptron formula `y = f(Σ(wi·xi) + b)` is the **same line equation** from Lesson 3, just generalized:

```
Lesson 3:  y = w·x + b                          (1 feature)
Lesson 4:  y = w1·x1 + w2·x2 + b                (2 features)
Lesson 7:  y = Σ(wi·xi) + b                      (N features)
```

The Σ (sum) is just shorthand for "add up all w·x pairs." It's the same as `X @ W` (matrix multiplication from Lesson 1) and `np.dot(X, weights)` (from Lessons 4-5). Three notations, one operation.

---

#### Activation Function: The Only New Part

The perceptron adds an **activation function** after the linear part. Different lessons used different activations:

```
       ┌─────────────┐     ┌────────────────────┐
Inputs → │ Σ(wi·xi) + b │ ──→ │ Activation function │ ──→ Output
       └─────────────┘     └────────────────────┘
        linear part          nonlinear part
       (same everywhere)     (changes per task)
```

| Lesson | Linear part | Activation | Output |
|--------|------------|------------|--------|
| Lesson 3 (linear reg.) | `w·x + b` | none (identity) | any number |
| Lesson 4 (logistic reg.) | `X @ w + b` | sigmoid (smooth 0→1) | probability |
| Lesson 7 (perceptron) | `Σ(wi·xi) + b` | step (hard 0 or 1) | binary decision |
| Neural nets | `X @ W + b` | ReLU / sigmoid / ... | depends on layer |

**Step vs sigmoid:**
- Sigmoid = dimmer (smooth transition 0→1)
- Step = light switch (instant jump: off or on, nothing in between)

---

#### Why Step Function Is a Problem

Step function has **no useful gradient** (derivative = 0 everywhere except at z=0). So the perceptron can't use gradient descent — it uses a special update rule instead:

```python
# Perceptron rule (cell-7): update only when wrong
weights += learning_rate * error * xi

# Gradient descent (Lessons 3-5): update proportionally to gradient
weights -= learning_rate * gradient
```

Replacing step with sigmoid is what made gradient descent possible and opened the door to deep neural networks.

---

#### Perceptron Rule vs Gradient Descent

Both find weights and bias, but **how** they update them is different:

| | Perceptron rule | Gradient descent |
|--|----------------|-----------------|
| **When it updates** | Only on mistakes | Every iteration |
| **Error magnitude** | Doesn't matter (wrong = wrong) | Matters (wrong by 0.01 vs 0.9) |
| **Prediction type** | 0 or 1 (step) | 0.0 ... 1.0 (sigmoid) |
| **Loss function** | None formally | MSE or BCE |

**Analogy:**
- Perceptron rule = teacher who says "right" or "wrong" (no detail)
- Gradient descent = teacher who says "colder/warmer" and by how much

**Why perceptron can't use gradient descent:** step function has derivative = 0 everywhere → gradient = 0 → weights never update. Perceptron works around this with a special rule: "if wrong, nudge weights toward the correct answer."

**Sigmoid fixed this:** smooth derivative → gradient descent works → can train many layers → deep learning.

```
1957: Perceptron (step + perceptron rule) → dead end for deep nets
1986: Backpropagation (sigmoid + gradient descent) → multi-layer networks
2012+: Deep Learning (ReLU + gradient descent + GPU) → modern AI
```

In modern ML, **only gradient descent** is used (with smooth activation functions).

---

#### Logic Gates: Simplest Classification Tasks

AND and OR are the easiest test for a perceptron — just 4 data points, 2 classes:

- **AND** (both must be 1): like needing passport AND ticket to enter
- **OR** (at least one must be 1): like buying coffee if tired OR cold

The perceptron learns a line separating 0s from 1s, just like logistic regression. Logic gates prove the perceptron works. XOR proves its **limit**.

---

#### XOR: Why One Perceptron Isn't Enough

A single perceptron = one line. AND and OR can be split by one line. XOR cannot:

```
AND (solvable):        OR (solvable):        XOR (impossible!):

  1│ 0    1             1│ 1  ╱ 1             1│ 1    0
   │   ╱                 │  ╱                   │
   │ ╱                   │╱                     │    ???
  0│0    0              0│0    1              0│ 0    1
   └──────               └──────               └──────

One line works ✓        One line works ✓      No single line works ✗
```

**Solution:** Stack multiple perceptrons → multi-layer neural network (next lesson). Two lines can solve what one cannot.

---

#### NLP Example: Hand-Picked Features vs Automatic (cell-17/18)

The word classifier in this lesson uses **manually chosen features** — not BoW:

```python
word_features("running") → [7, 2, 1, 0]
#                           length, vowels, ends_ing, ends_ed

word_features("cat")     → [3, 1, 0, 0]
```

The perceptron then classifies: `step(w1×length + w2×vowels + w3×ing + w4×ed + bias)` → verb or noun.

After training, the weights reveal what matters: `-ing` and `-ed` endings get high weights (strong verb signals), while length and vowels barely matter.

**Feature engineering vs automatic features:**

| Approach | Who picks features | Example |
|----------|-------------------|---------|
| **Feature engineering** (Lesson 7) | Human decides (length, endings...) | 4 numbers, hand-picked |
| **Bag of Words** (Lesson 5) | Automatic word counting | 45 numbers, no human knowledge |
| **Embeddings** (Module 5+) | Model learns features itself | 300 numbers, fully automatic |

Feature engineering was once the main job of ML engineers. Deep networks made it largely unnecessary — they learn useful features from raw data on their own.

---

*Notes from: Lesson 7 Q&A session*
