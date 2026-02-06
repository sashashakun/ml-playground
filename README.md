# ML Learning Curriculum: From Zero to Transformers

A hands-on, intuition-first curriculum for learning Machine Learning, focused on NLP examples and building toward understanding Transformers.

## Who Is This For?

- **Complete ML beginners** who prefer intuition over formulas
- **Developers** who learn best by building and experimenting
- **Anyone curious about NLP** and how language models work

## Philosophy

> "If you can't explain it simply, you don't understand it well enough."

This curriculum prioritizes:
1. **Intuition before formulas** - Understand *why* before *how*
2. **Visualizations** - See what's happening inside the models
3. **Building from scratch** - Then use the professional tools
4. **NLP focus** - Every concept connected to language/text

## Curriculum Overview

| Module | Topic | Analogy | Key Outcome |
|--------|-------|---------|-------------|
| 1 | Foundations | HTML + CSS | Think in matrices, understand loss & gradient descent |
| 2 | Classification | Vanilla JavaScript | Build a sentiment classifier |
| 3 | Neural Networks | jQuery | Implement a network from scratch |
| 4 | PyTorch | Backbone.js | Master the deep learning framework |
| 5 | Embeddings | React Components | Understand word vectors |
| 6 | Sequences (RNN/LSTM) | React + Redux | Generate text, see why attention is needed |
| 7 | Attention | React Hooks | Visualize what models "look at" |
| 8 | Transformers | Modern React | Build and use transformer models |

## Quick Start

### 1. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Jupyter Lab

```bash
jupyter lab
```

### 3. Start Learning

Open `notebooks/01_foundations/01_numpy_basics.ipynb` and follow along!

## Directory Structure

```
ml-playground/
├── notebooks/           # Interactive Jupyter notebooks (main learning path)
│   ├── 01_foundations/  # NumPy, data intuition, linear regression
│   ├── 02_classification/  # Logistic regression, sentiment analysis
│   ├── 03_neural_nets/  # Perceptrons, backprop from scratch
│   ├── 04_pytorch/      # Tensors, autograd, training loops
│   ├── 05_embeddings/   # Word2Vec, semantic similarity
│   ├── 06_sequences/    # RNNs, LSTMs, text generation
│   ├── 07_attention/    # Attention mechanism, visualization
│   └── 08_transformers/ # Self-attention, full transformer, Hugging Face
├── src/                 # Reference Python implementations
│   ├── models/          # Reusable model classes
│   ├── utils/           # Helper functions
│   └── data/            # Data loading utilities
├── data/                # Datasets (downloaded or created)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Learning Approach

For each notebook:

1. **Read & Run** - Follow along, execute each cell
2. **Experiment** - Change parameters, see what happens
3. **Break Things** - Understanding errors builds intuition
4. **Complete Projects** - Apply what you learned
5. **Explain It** - If you can teach it, you understand it

## Module Details

### Module 1: Foundations
*The building blocks of ML*

- Arrays and matrices (why they're everywhere in ML)
- What is "data"? Features, labels, distributions
- Linear regression: your first prediction
- **NLP Hook**: Representing words as numbers

### Module 2: Classification
*Making decisions with data*

- Logistic regression: from numbers to yes/no
- Sentiment analysis: is this review positive?
- scikit-learn: professional ML tools
- **Project**: Build a sentiment classifier

### Module 3: Neural Networks from Scratch
*Understanding the magic*

- Perceptrons: the simplest "brain cell"
- Stacking neurons: creating depth
- Backpropagation: how networks learn
- **Project**: Character-level text classifier

### Module 4: PyTorch Fundamentals
*Framework thinking*

- Tensors: supercharged arrays
- Autograd: automatic differentiation magic
- Training loops: the standard ML workflow
- **Project**: Rebuild your classifier in PyTorch

### Module 5: Embeddings
*Words have meaning*

- From one-hot to dense vectors
- Word2Vec: "King - Man + Woman = Queen"
- Visualizing word relationships
- **Project**: Explore word similarities

### Module 6: Sequence Models
*Remembering the past*

- Why order matters in language
- RNNs: networks with memory
- LSTMs: solving the forgetting problem
- **Project**: Generate text in any style
- **Critical Insight**: Why RNNs struggle with long texts

### Module 7: Attention
*The breakthrough*

- "Which words should I focus on?"
- Query, Key, Value (like a search engine)
- Attention visualization
- **Project**: Translator with attention

### Module 8: Transformers
*The revolution*

- Self-attention: words understanding each other
- Positional encoding: keeping track of order
- Multi-head attention: multiple perspectives
- BERT vs GPT: understanding vs generating
- **Project**: Fine-tune a transformer for your task

## Prerequisites

- **Python basics** - variables, functions, loops, classes
- **Some math comfort** - but we explain everything visually
- **Curiosity** - the most important ingredient!

## Tips for Success

- **Don't skip modules** - each builds on the previous
- **Run every cell** - even if you think you understand
- **Read error messages** - they teach you a lot
- **Take breaks** - learning takes time to consolidate
- **Ask "why?"** - not just "how?"

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Course](https://huggingface.co/course)
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

*Happy learning! Remember: everyone starts at zero.*
