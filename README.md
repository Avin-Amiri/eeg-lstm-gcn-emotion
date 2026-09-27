# Temporal–Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM–GCN Architecture

## Abstract

This repository provides a two-stage deep-learning pipeline for three-class emotion recognition from differential entropy (DE) features in the SEED EEG dataset. In the first stage, channel-wise bidirectional LSTM encoders transform five-band features into electrode-level representations. In the second stage, a graph convolutional network (GCN) aggregates those representations over an EEG electrode graph and predicts an emotion class. The implementation includes feature preprocessing, graph construction, cached embeddings, a stratified evaluation split, and a grid search over selected GCN configurations.

## Keywords

EEG; emotion recognition; SEED; differential entropy; LSTM; graph convolutional network; deep learning

## 1. Introduction

Electroencephalography (EEG) provides temporally rich signals for studying affective states. This project combines recurrent feature encoding with graph-based spatial aggregation: channel-wise LSTMs encode the input features for each electrode, and a GCN models relationships between electrodes in a fixed graph. The pipeline uses the pre-extracted SEED features and is implemented in [`seed_lstm_gcn.py`](seed_lstm_gcn.py).

## 2. Repository Structure

```text
.
├── seed_lstm_gcn.py       # Data loading, preprocessing, model training, and grid search
├── channel-order.xlsx     # Electrode order used to map channel names to graph nodes
├── requirements.txt       # Project dependencies
├── .gitignore
└── README.md
```

## 3. Methodology

### 3.1. Dataset and Input Features

The pipeline expects the SEED `ExtractedFeatures_4s` MATLAB feature files. Each trial variable is expected to have shape `(62, 5, time_steps)`, where the dimensions correspond to 62 EEG channels, five DE frequency-band features, and extracted time steps. The loader converts these arrays to samples of shape `(samples, 62, 5)`. The five features correspond to the frequency bands provided in the extracted-feature dataset.

Trial variables are associated with the three integer class IDs (`0`, `1`, and `2`) using the fixed trial-label sequence in the script. Refer to that sequence when interpreting class-specific results.

### 3.2. Electrode Graph

The implementation constructs a fixed, undirected graph from electrode connections defined in `seed_lstm_gcn.py`. The first column of the channel-order spreadsheet maps the feature-array channel order to graph node indices. Channel names are normalized to uppercase and whitespace is trimmed; only connections whose endpoint names are present in the spreadsheet are included. The graph is cached as `edge_index.pt`.

### 3.3. Two-Stage LSTM–GCN Framework

```text
+--------------------------------------------------------------+
| Input DE features: one sample with 62 channels x 5 features  |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 1: Channel-wise Bi-LSTM feature encoding               |
| - One encoder is trained independently for each channel      |
| - Input to each encoder: (batch, 1, 5)                       |
| - Embedding dimensions searched: 16, 32, and 64              |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 2: Graph convolutional classification                  |
| - Nodes: EEG electrodes; node features: LSTM embeddings       |
| - Edges: fixed electrode adjacency graph                      |
| - GCN layers followed by global mean pooling and classifier   |
+--------------------------------------------------------------+
```

The LSTM encoders use two bidirectional layers, with a learned projection to the selected embedding dimension. The GCN search varies embedding dimension, hidden dimension, and number of graph-convolution layers.

## 4. Installation

Use Python 3.9 or later. Install the Python dependencies and a PyTorch/PyTorch Geometric build compatible with your platform and CUDA setup (if applicable):

```bash
pip install numpy pandas scipy scikit-learn openpyxl
pip install torch
pip install torch-geometric
```

## 5. Data Preparation and Execution

Prepare the extracted SEED MATLAB files and an Excel channel-order file. The channel-order file should have one electrode name per row in its first column, following the same order as the 62 channels in the feature arrays.

Run the pipeline with paths appropriate for your environment:

```bash
python seed_lstm_gcn.py \
  --data_dir "/path/to/ExtractedFeatures_4s" \
  --channel_path "/path/to/channel-order.xlsx" \
  --save_dir "saved_features"
```

All three command-line arguments are optional. The script defaults to a Windows-specific SEED data directory, `channel-order.xlsx`, and `saved_features`. The script uses CUDA when available and otherwise runs on CPU.

## 6. Training and Evaluation

The pipeline uses a stratified 70%/15%/15% train/validation/test split with split seed 42. For each embedding dimension, it evaluates the following GCN configuration grid:

| Parameter | Values |
|---|---|
| LSTM embedding dimension | 16, 32, 64 |
| GCN hidden dimension | 32, 64, 128 |
| Number of GCN layers | 2, 3 |

GCN checkpoints are selected by validation loss. Test accuracy for each configuration is written to the results CSV. The random seeds for NumPy and PyTorch are set to 42; some GPU operations may remain nondeterministic.

## 7. Outputs

The `--save_dir` folder is created automatically. Depending on the run and available caches, it contains:

- `X_raw.npy`, `y_raw.npy` — standardized input samples and integer class labels.
- `edge_index.pt` — cached graph connectivity.
- `embeddings_dim16.npy`, `embeddings_dim32.npy`, `embeddings_dim64.npy` — cached channel embeddings.
- `model_emb*_hid*_lay*.pth` — GCN checkpoints selected using validation loss.
- `grid_search_results.csv` — test accuracy for the evaluated configurations.

Cached arrays, embeddings, and graph connectivity are reused when present in `--save_dir`. Use a separate output directory or remove the relevant generated cache files after changing source data or channel ordering.

## 8. Implementation Details

The implementation operates on pre-extracted DE features, rather than raw EEG recordings. Each channel encoder receives a single five-feature input step and produces a channel embedding; the subsequent GCN performs spatial aggregation over the electrode graph. The executable script is the reference for the implemented preprocessing, label sequence, model configuration, and command-line options.

## Citation

If you use this implementation or describe the associated framework in your work, cite the paper:

```bibtex
@article{amiri2025temporalSpatialLstmGcn,
  title   = {Temporal--Spatial Graph-Integrated Framework for EEG-Based Emotion Recognition Using LSTM--GCN Architecture},
  author  = {Amiri, Zahra and Mohseni, Abdorreza Hesam},
  year    = {2025}
}
```

## License

No license file is included in this repository. Please contact the authors for permission before redistributing or reusing the code beyond applicable copyright exceptions.

---

Paper authors: Zahra Amiri and Abdorreza Hesam Mohseni; University of Guilan.

Dataset: [SEED: SJTU Emotion EEG Dataset](https://bcmi.sjtu.edu.cn/~seed/seed.html).

## Acknowledgements

The authors acknowledge the SEED dataset and its contributors for making EEG emotion-recognition research possible.

## Project Structure at a Glance

```text
SEED extracted features (.mat)
               |
               v
       Feature loading and scaling
               |
               v
     Channel-wise LSTM embeddings
               |
               v
 Electrode graph + GCN classifier
               |
               v
     Grid-search results and models
```
