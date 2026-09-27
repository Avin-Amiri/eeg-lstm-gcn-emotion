# Temporal-Spatial Graph-Integrated Framework for EEG Emotion Recognition (LSTM-GCN)

This repository contains the official PyTorch implementation of the dual-stage LSTM-GCN pipeline for subject-dependent/independent EEG emotion recognition evaluated on the SEED benchmark dataset.

The architecture decouples temporal dynamics and spatial electrode topology into two sequential stages:
1. **Channel-wise Temporal Representation:** A bidirectional LSTM network extracts temporal dynamics independently for each of the 62 EEG channels from differential entropy (DE) features across 5 frequency bands ($\delta, \theta, \alpha, \beta, \gamma$).
2. **Spatial Topology Modeling:** A Graph Convolutional Network (GCN) processes the learned channel embeddings over an anatomical EEG adjacency graph to perform 3-class emotion classification (Positive, Neutral, Negative).


## Repository Structure

├── seed_lstm_gcn.py        # Main execution script (data loading, featurization, grid search)
├── channel-order.xlsx      # Electrode layout definition (62 channels)
├── requirements.txt        # Core dependencies
├── .gitignore
└── README.md


## Method Overview

Input DE Features (62 channels x 5 bands)
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: Channel-wise Bi-LSTM Feature Extraction      │
│  - 62 independent models (one per electrode)           │
│  - Maps raw band inputs into temporal embedding space  │
│  - Dimension: d_embed ∈ {16, 32, 64}                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 2: Spatial GCN Topological Aggregation           │
│  - Fixed graph structure constructed from 10-20 layout │
│  - Node features: LSTM channel embeddings              │
│  - Message passing across adjacent electrodes          │
│  - Global mean pooling + Linear classification         │
└────────────────────────────────────────────────────────┘


## Dataset & Preprocessing

- **Dataset:** [SEED (SJTU Emotion EEG Dataset)](https://bcmi.sjtu.edu.cn/~seed/seed.html)
- **Features:** Pre-computed Differential Entropy (`ExtractedFeatures_4s`) with 5 frequency bands per channel.
- **Labels:** 3 classes mapped across 15 movie clips per session:
  - `0`: Negative
  - `1`: Neutral
  - `2`: Positive
- **Graph Topology:** Derived from standard 10–20 EEG electrode placement, encoded via bilateral connections across adjacent nodes.

## Installation

Ensure CUDA 11.8+ or 12.x is available if running with GPU acceleration.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install PyTorch matching your CUDA version, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

## Usage

### 1. Prepare Directory Structure
Place the SEED dataset and the channel configuration file in accessible directories:
- SEED `.mat` files in `data/SEED_EEG/ExtractedFeatures_4s` (or specify via CLI).
- `channel-order.xlsx` in the repository root.

### 2. Execution
To run the complete pipeline (data loading, stage-1 temporal training, stage-2 GCN hyperparameter grid search):


python seed_lstm_gcn.py \
    --data_dir "data/ExtractedFeatures_4s" \
    --channel_path "channel-order.xlsx" \
    --save_dir "saved_features" \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001


### Script Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | `data/ExtractedFeatures_4s` | Directory containing SEED .mat feature files |
| `--channel_path` | `str` | `channel-order.xlsx` | Path to electrode channel order file |
| `--save_dir` | `str` | `saved_features` | Directory to cache intermediate embeddings and weights |
| `--epochs` | `int` | `50` | Maximum training epochs for the spatial GCN |
| `--batch_size` | `int` | `64` | Batch size for GCN training |
| `--lr` | `float` | `1e-3` | Learning rate for Adam optimizer |


## Hyperparameter Grid Search

The pipeline performs automated validation across the following parameter grid for the spatial classifier:
- **LSTM Embedding Dimensions:** `[16, 32, 64]`
- **GCN Hidden Channels:** `[32, 64, 128]`
- **GCN Layers:** `[2, 3]`

Results, checkpoints, and evaluation metrics (Accuracy, Macro-F1) are logged directly to `saved_features/grid_search_results.csv`.


## Citation

If you use this codebase or architecture in your research, please cite:

@article{amiri2025lstm_gcn_eeg,
  title={Temporal--Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM--GCN Architecture},
  author={Amiri, Zahra and Mohseni, Abdorreza Hesam},
  year={2025}
}
