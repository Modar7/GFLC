# GFLC: Graph-based Fairness-aware Label Correction 
# Accepted at Baltic Journal of Modern Computing


> ⚠️ **Work in Progress**: This repository is currently under development. Updates and improvements will be added in the near future.

## Overview

This repository contains the official implementation of **GFLC (Graph-based Fairness-aware Label Correction)**, a novel method for correcting instance-dependent label noise while preserving demographic parity in machine learning datasets.

GFLC addresses the critical challenge of learning with noisy labels in the context of fairness-aware machine learning. Our approach combines three key components:
- **Prediction confidence measure** for identifying uncertain predictions
- **Graph-based regularization** through Ricci-flow-optimized graph Laplacians
- **Explicit demographic parity incentives** for fair label correction

## Key Features

- **Graph-based approach**: Utilizes k-nearest neighbor graphs with Forman-Ricci curvature and discrete Ricci flow
- **Fairness-aware**: Explicitly optimizes for demographic parity during label correction
- **Instance-dependent noise handling**: Specifically designed for complex noise patterns that depend on both features and sensitive attributes
- **GPU acceleration**: Supports CUDA for efficient processing of large datasets
- **Flexible configuration**: Adjustable parameters for different noise rates and fairness requirements

## Paper

**Title**: GFLC: Graph-based Fairness-aware Label Correction for Fair Classification

**ArXiv**: [https://arxiv.org/pdf/2506.15620](https://arxiv.org/pdf/2506.15620)

**Abstract**: Fairness in machine learning has critical importance for building trustworthy systems as AI increasingly impacts various aspects of society. However, training data often contains biased and noisy labels that affect both model performance and fairness. GFLC presents an efficient method for correcting label noise while preserving demographic parity through graph-based techniques, prediction confidence measures, and explicit fairness incentives.

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn networkx scipy torch lightgbm
```

### Dependencies

For faster k-NN computation on large datasets:
```bash
pip install pynndescent
```

For dataset handling (following Fair-OBNC setup):
```bash
pip install aequitas
```

## Dataset

The experiments use the Bank Account Fraud dataset (Variant II). The dataset can be downloaded following the same procedure as described in [Fair-OBNC](https://github.com/feedzai/fair-obnc/tree/main):


## Usage

### Basic Usage

```python
from GFLC import GFLC
import numpy as np
import pandas as pd

# Initialize GFLC
gflc = GFLC(
    k=10,              # Number of nearest neighbors
    ricci_iter=2,      # Number of Ricci flow iterations
    alpha=0.3,         # Weight for margin term
    beta=0.5,          # Weight for graph Laplacian term
    gamma=0.2,         # Weight for fairness term
    pos_threshold=0.25, # Threshold for positive predictions
    neg_threshold=0.85, # Threshold for negative predictions
    max_fpr=0.03       # Maximum false positive rate
)

# Fit the model
gflc.fit(X_train, y_train, s_train)

# Correct labels
y_corrected = gflc.correct_labels(
    X_train, y_train, s_train,
    disparity_target=0.05  # Target disparity level
)
```

## Parameters

### GFLC Constructor Parameters

- `k` (int, default=10): Number of nearest neighbors for graph construction
- `ricci_iter` (int, default=2): Number of Ricci flow iterations for graph optimization
- `alpha` (float, default=0.2): Weight for prediction confidence (margin) term
- `beta` (float, default=0.6): Weight for graph Laplacian regularization term
- `gamma` (float, default=0.2): Weight for fairness penalty term
- `n_jobs` (int, default=-1): Number of parallel jobs for computations
- `approx_knn` (bool, default=True): Use approximate k-NN for large datasets
- `sample_size` (int, default=None): Subsample size for very large datasets


## Method Components

### 1. Graph Construction
GFLC builds a k-nearest neighbor graph using inverse distance weighting:
```
w_ij = 1 / (distance(i,j) + ε)
```

### 2. Forman-Ricci Curvature
Computes discrete Forman-Ricci curvature for each edge to measure local connectivity patterns.

### 3. Ricci Flow
Updates edge weights based on curvature to enhance graph geometric properties:
- Positive curvature edges (well-connected regions) get strengthened
- Negative curvature edges (bottlenecks) get weakened

### 4. Combined Scoring Function
The correction score combines three terms:
```
score(i) = α × margin_term(i) + β × laplacian_term(i) + γ × fairness_term(i)
```

## File Structure

```
├── GFLC.py          # Main GFLC implementation
├── run.py           # Example usage and experimental setup
├── README.md        # This file
```

## License & copyright
Licensed under the [MIT License](License).

## Contact
For questions or issues, please contact [modar.sulaiman@ut.ee].

