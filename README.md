# Temporal–Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM–GCN Architecture

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper:  
**"Temporal–Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM–GCN Architecture"**  
*Zahra Amiri, Abdorreza Hesam Mohseni*  
*Proceedings of the 6th International Conference on Soft Computing (ICSC 2025)*

*https://civilica.com/doc/2720860*
---

## 📌 Overview

Recognizing emotional states from electroencephalogram (EEG) signals requires modeling both temporal signal dynamics and non-Euclidean spatial functional dependencies among distributed cortical electrodes. 

This repository implements a **dual-stage, temporal-spatial deep learning pipeline**:
1. **Stage 1 (Channel-wise Dynamic Temporal Encoding):** Independent, multi-layer LSTMs are trained per electrode to project 5-frequency Differential Entropy (DE) features into rich latent node representations ($\mathbf{Z} \in \mathbb{R}^{N \times 62 \times d_{\text{emb}}}$).
2. **Stage 2 (Topological Spatial Graph Convolution):** A Graph Convolutional Network (GCN) models topological inter-channel dependencies over an anatomically defined 62-channel 10–20 electrode graph (108 bidirectional / 216 directed edges), followed by global channel pooling and multi-class emotion classification.

Tested on the **SEED (SJTU Emotion EEG Dataset)** benchmark across 3 emotion classes (*Negative*, *Neutral*, *Positive*), this architecture achieves a peak test accuracy of **97.98%** and an F1-score of **97.98%**.

---

## 🏛 Framework Architecture

```
Raw EEG (.mat) ──► Differential Entropy (5 bands) ──► Shape: (N, 62, 5)
                                                             │
┌────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┐
│ Stage 1: Channel-Wise Temporal Node Pre-training (LSTM)                                                                 │
│   Each channel c ∈ {1, ..., 62} is processed by an independent 2-layer LSTM:                                            │
│       h_t^(c) = LSTM(x_t^(c), h_(t-1)^(c))  ==> Node Embedding Matrix Z ∈ R^(N × 62 × d_emb)                            │
└────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                                             │
┌────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┐
│ Stage 2: Spatial Brain Topology Graph Learning (GCN)                                                                    │
│   Electrode Graph G = (V, E): |V| = 62, |E| = 216 directed edges (derived from 10-20 system)                            │
│                                                                                                                         │
│   Spectral Graph Convolution Layer:                                                                                     │
│       H^(l+1) = LeakyReLU( BatchNorm( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) ) )                                             │
│                                                                                                                         │
│   Global Graph Pooling & Classification:                                                                                │
│       h_graph = (1 / |V|) ∑_{i=1}^{|V|} H_i^(L)                                                                        │
│       ŷ = Linear( ReLU( Dropout( Linear(h_graph) ) ) )  ==> Softmax Logits (3 Classes)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Experimental Results

Evaluated on stratified SEED splits (70% train, 15% validation, 15% test). Early stopping (patience = 20) was guided by validation accuracy.

### 1. Grid Search Ablation Study (18 Configurations)

| Rank | Embedding Dim ($d_{\text{emb}}$) | Hidden Dim ($d_{\text{hidden}}$) | GCN Layers ($L$) | Test Accuracy (%) | Weighted F1 (%) | test Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **64** | **256** | **3** | **97.98%** | **97.98%** | **0.0558** |
| 🥈 2 | 128 | 256 | 3 | 97.84% | 97.83% | 0.0766 |
| 🥉 3 | 32 | 256 | 3 | 97.20% | 97.20% | 0.1153 |

### 2. Comparison with Benchmark Baselines on SEED

| Method | Feature / Architecture | Accuracy (%) |
|---|---|:---:|
| SVM Baseline (2019) | Differential Entropy | 56.70% |
| Attention Network (2025) | Temporal Attention | 79.30% |
| Vanilla GCN (Kipf & Welling, 2017) | Raw Electrode Graph | 81.56% |
| GMSS (2022) | Multi-Source Domain Generalization | 86.52% |
| **Proposed Framework (Ours)** | **Temporal LSTM + Spatial GCN** | **97.98%** |

---

## 📁 Repository Structure

```text
├── data/
│   ├── ExtractedFeatures_4s/      # SEED .mat feature files (DE features: 62 channels × 5 bands)
│   ├── channel-order.xlsx         # 10-20 standard electrode order (62 channels)
│   └── saved_features/            # Preprocessed arrays, graph caches, and checkpoints
│       ├── X_raw.npy              # Cached EEG feature tensors
│       ├── y_raw.npy              # Ground truth emotion labels (0: Neg, 1: Neu, 2: Pos)
│       ├── edge_index.pt          # PyG graph connectivity tensor: shape (2, 216)
│       ├── embeddings/            # Extracted node embedding matrices (dim = 32, 64, 128)
│       └── models/                # Saved model weights (.pth)
├── main.py                        # Full end-to-end training and evaluation script
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git ignore rules for checkpoints and large arrays
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone & Virtual Environment

```bash
git clone https://github.com/<username>/eeg-lstm-gcn.git
cd eeg-lstm-gcn

python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 2. Install PyTorch & Dependencies

Install PyTorch according to your local CUDA driver (recommended: PyTorch $\ge 2.1$ with CUDA 11.8 or 12.1):

```bash
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric:
pip install torch-geometric

# Install remaining requirements:
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### Step 1: Data Preparation
Ensure the SEED extracted differential entropy features and channel layout are placed inside the `data/` directory:
- `data/ExtractedFeatures_4s/*.mat`
- `data/channel-order.xlsx`

### Step 2: Run End-to-End Pipeline
The pipeline handles preprocessing, channel graph construction, Stage-1 node embedding pretraining, and Stage-2 GCN grid search:

```bash
python seed_lstm_gcn.py
```

### Step 3: Outputs & Reproducibility
- Intermediate node embeddings are saved to `data/saved_features/embeddings/embeddings_dim{32,64,128}.npy`.
- Best weights for each hyperparameter configuration are saved to `data/saved_features/models/gcn_emb{E}_hidden{H}_layers{L}.pth`.
- Comprehensive evaluation metrics are saved to `data/saved_features/grid_search_results.csv`.

---

## 🔬 Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| Input Dimension | 5 | Differential Entropy features ($\delta, \theta, \alpha, \beta, \gamma$) |
| Electrode Nodes ($N$) | 62 | Standard 10–20 electrode configuration |
| Graph Directed Edges ($E$) | 216 | 108 bidirectional anatomical adjacencies |
| Optimizer | AdamW | Stage 2 GCN (`weight_decay = 1e-4`) |
| Learning Rate | $1 \times 10^{-3}$ | Reduced by 0.5 via `ReduceLROnPlateau` (patience = 10) |
| Batch Size | 32 | Mini-batch sample size |
| Max Epochs | 100 | Stage 2 GCN |
| Early Stopping | 20 | Epochs on validation accuracy |
| Gradient Clipping | 1.0 | Max $L_2$-norm clipping |


---

## 📖 Citation

If you use this codebase or framework in your research, please cite our conference paper:

```bibtex
@inproceedings{amiri2025temporal,
  title={Temporal--Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM--GCN Architecture},
  author={Amiri, Zahra and Mohseni, Abdorreza Hesam},
  booktitle={Proceedings of the 6th International Conference on Soft Computing (ICSC 2025)},
  year={2025},
  month={November},
  address={Rasht, Iran},
  organization={Faculty of Technology and Engineering, University of Guilan}
}
```

---

## 📬 Contact & Inquiries

For technical questions or prospective research discussions, please contact:
- **Avin Amiri** — [zahraamiri@khu.ac.ir](mailto:zahraamiri@khu.ac.ir)
