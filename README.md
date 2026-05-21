# logistic-regression-from-scratch
Production-grade Logistic Regression implementation from scratch using NumPy with optimization analysis, numerical stability engineering, regularization, and sklearn benchmarking.
# logistic-regression-from-scratch

Production-grade Logistic Regression implementation from scratch using NumPy with optimization analysis, numerical stability engineering, regularization, and sklearn benchmarking.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?logo=numpy)](https://numpy.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project implements Logistic Regression entirely from scratch using NumPy without relying on sklearn’s built-in `LogisticRegression`.

The project focuses on:

- Optimization behavior
- Numerical stability
- Regularization
- Threshold analysis
- Convergence dynamics
- Benchmarking against sklearn

The model is evaluated on a Loan Approval Prediction dataset using a complete preprocessing and experimentation pipeline.

---

## Features

- Logistic Regression from scratch
- Mini-batch Gradient Descent
- Momentum Optimization
- Adam Optimizer
- Cosine Learning Rate Scheduling
- L2 Regularization
- Stable Sigmoid Implementation
- Numerically Stable BCE Loss
- Gradient Clipping
- One-Hot Encoding
- Z-Score Normalization
- Threshold Optimization
- Feature Importance Analysis
- sklearn Benchmarking

---

## Optimization Methods

The project compares multiple optimization approaches:

| Optimizer | Description |
|---|---|
| SGD | Standard mini-batch gradient descent |
| Momentum | Accelerated optimization using velocity updates |
| Adam | Adaptive moment estimation |
| Adam + Cosine Scheduling | Adaptive optimization with dynamic learning-rate scheduling |

---

## Numerical Stability Engineering

Several techniques were implemented to improve training reliability:

```python
def stable_sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

Additional engineering improvements:
- Gradient clipping
- Stable BCE-from-logits formulation
- Overflow protection
- Constant-feature handling during normalization

---

## Results

| Model | Validation Accuracy | Notes |
|---|---|---|
| SGD | Competitive | Slower convergence |
| Momentum | Improved stability | Faster optimization |
| Adam | Best convergence | Most stable training |
| sklearn LogisticRegression | Reference benchmark | L-BFGS optimizer |

The from-scratch implementation achieved performance close to sklearn while providing full control over optimization dynamics and experimentation.

---

## Visualizations

The notebook includes:

- Loss Curves
- Convergence Analysis
- Gradient Norm Tracking
- ROC Curves
- Confusion Matrix
- Feature Importance Plots
- Threshold Sensitivity Analysis
- Regularization Path Analysis

---

## Repository Structure

```text
logistic-regression-from-scratch/
├── notebook/
│   └── logistic_regression_from_scratch.ipynb
├── report/
│   └── logistic_regression_technical_report.pdf
├── results/
│   ├── figures/
│   │   ├── optimizer_comparison.png
│   │   ├── roc_curve.png
│   │   ├── feature_importance.png
│   │   └── threshold_analysis.png
│   └── metrics_summary.txt
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup & Usage

```bash
pip install numpy pandas matplotlib scikit-learn
```

Run the notebook:

```bash
jupyter notebook logistic_regression_from_scratch.ipynb
```

---

## Dataset

Loan Approval Prediction Dataset containing:
- financial attributes
- demographic information
- loan status labels

The dataset is split into:
- 70% training
- 15% validation
- 15% testing

using stratified sampling.

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Scikit-learn` · `Jupyter Notebook`

---

## Key Findings

- Adam optimizer achieved the fastest and most stable convergence
- Numerical stabilization significantly improved training reliability
- Threshold tuning improved precision-recall tradeoffs
- Moderate regularization improved generalization performance
- From-scratch implementation achieved performance close to sklearn

---

## Future Improvements

Potential future extensions include:

- RMSProp
- Newton’s Method
- Early Stopping
- Multiclass Softmax Regression
- GPU Acceleration
- Interactive Experiment Dashboard
- 
