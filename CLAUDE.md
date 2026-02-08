# ML Playground - Project Context

## What This Is
A hands-on ML learning curriculum with 27 Jupyter notebooks, progressing from NumPy basics to Transformers.

## Current Learning Progress
- ✅ **Lesson 1**: NumPy basics, matrix multiplication, word vectors, one-hot encoding
- ✅ **Lesson 2**: Data intuition, features/labels, normalization, train/test split, bag-of-words
- 🔜 **Lesson 3**: Linear regression (next up)

## Key Files
- `MY_NOTES.md` - Personal learning notes with Q&A insights (synced to GitHub)
- `notebooks/` - 27 Jupyter notebooks organized by module (01-08)
- `requirements.txt` - Dependencies (use with venv)
- `README.md` - Has Colab badges for mobile learning

## Setup
```bash
source venv/bin/activate
# VS Code: Select "ML Playground (venv)" kernel
```

## User Preferences
- Prefers intuition and visualizations over formulas
- Learning style: Ask questions, get explanations, then "add to notes"
- Uses VS Code locally + Claude mobile app for theory review
- GitHub repo: https://github.com/sashashakun/ml-playground

## Recent Fixes Applied
- Lesson 2, cell-11: Fixed scatter plot to use `data` directly (avoids X/y mismatch)
- Lesson 2, cell-17: Fixed `value_counts()` to work with numpy arrays
- Added "Run all cells" warning for Colab users

## Workflow
1. User runs notebooks locally in VS Code
2. Asks Claude questions about concepts
3. Claude explains with intuition-first approach
4. User says "add to notes" → Claude updates MY_NOTES.md
5. Changes get committed and pushed to GitHub
