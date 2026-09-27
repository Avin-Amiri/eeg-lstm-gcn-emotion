import os
import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")


# ==========================================
# 0. REPRODUCIBILITY & ARGUMENTS
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatial-Temporal Graph-Integrated Framework for SEED EEG Emotion Recognition"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\dataset\SEED\SEED_EEG\ExtractedFeatures_4s",
        help="Path to SEED ExtractedFeatures_4s directory containing .mat files",
    )
    parser.add_argument(
        "--channel_path",
        type=str,
        default="channel-order.xlsx",
        help="Path to channel-order.xlsx coordinate/order file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_features",
        help="Directory to cache intermediate numpy arrays, graph edges, and models",
    )
    return parser.parse_args()


# ==========================================
# 1. DATA LOADING & PREPROCESSING (SEED)
# ==========================================
def load_eeg_data_corrected(data_dir, save_dir):
    """
    Loads SEED extracted features (.mat), reshapes trial data into temporal samples,
    and caches them as numpy arrays.
    """
    raw_x_path = os.path.join(save_dir, "X_raw.npy")
    raw_y_path = os.path.join(save_dir, "y_raw.npy")

    if os.path.exists(raw_x_path) and os.path.exists(raw_y_path):
        print(f"[Info] Loading cached raw data from '{save_dir}'...")
        X = np.load(raw_x_path)
        y = np.load(raw_y_path)
        return X, y

    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    session_labels = np.array([2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0])

    X_list = []
    y_list = []

    print("[Info] Parsing SEED .mat files...")
    for f in mat_files:
        path = os.path.join(data_dir, f)
        mat = sio.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]

        for key in keys:
            match = re.search(r"(\d+)$", key)
            if not match:
                continue
            idx = int(match.group(1)) - 1
            if idx >= 15:
                continue

            arr = mat[key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 3:
                continue

            # Target shape: (N, 62, 5, time_steps)
            if arr.shape[0] != 62 or arr.shape[1] != 5:
                continue

            # (62, 5, T) -> (T, 62, 5)
            arr = arr.transpose(2, 0, 1)

            X_list.append(arr)
            y_list.append(np.full(arr.shape[0], session_labels[idx]))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Standardize across feature dimension
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - mean) / std

    np.save(raw_x_path, X)
    np.save(raw_y_path, y)
    print(f"[Info] Cached processed data: X shape={X.shape}, y shape={y.shape}")
    return X, y


# ==========================================
# 2. ELECTRODE GRAPH TOPOLOGY
# ==========================================
def build_graph(channel_path, save_dir):
    """
    Constructs an undirected electrode topology graph (bidirectional edge_index)
    based on standard 10-20 EEG adjacent electrode pairings.
    """
    edge_index_path = os.path.join(save_dir, "edge_index.pt")
    if os.path.exists(edge_index_path):
        print(f"[Info] Loading cached graph topology from '{edge_index_path}'...")
        return torch.load(edge_index_path)

    df = pd.read_excel(channel_path, header=None)
    channels = [str(c).strip().upper() for c in df[0].values]
    ch2idx = {c: i for i, c in enumerate(channels)}

    # Standard physical/functional EEG electrode connections
    pairs = [
        ("FP1", "FPZ"), ("FPZ", "FP2"), ("FP1", "AF3"), ("FP2", "AF4"),
        ("AF3", "F1"),  ("AF4", "F2"),  ("F7", "F5"),   ("F5", "F3"),
        ("F3", "F1"),   ("F1", "FZ"),   ("FZ", "F2"),   ("F2", "F4"),
        ("F4", "F6"),   ("F6", "F8"),   ("FT7", "FC5"), ("FC5", "FC3"),
        ("FC3", "FC1"), ("FC1", "FCZ"), ("FCZ", "FC2"), ("FC2", "FC4"),
        ("FC4", "FC6"), ("FC6", "FT8"), ("T7", "C5"),   ("C5", "C3"),
        ("C3", "C1"),   ("C1", "CZ"),   ("CZ", "C2"),   ("C2", "C4"),
        ("C4", "C6"),   ("C6", "T8"),   ("TP7", "CP5"), ("CP5", "CP3"),
        ("CP3", "CP1"), ("CP1", "CPZ"), ("CPZ", "CP2"), ("CP2", "CP4"),
        ("CP4", "CP6"), ("CP6", "TP8"), ("P7", "P5"),   ("P5", "P3"),
        ("P3", "P1"),   ("P1", "PZ"),   ("PZ", "P2"),   ("P2", "P4"),
        ("P4", "P6"),   ("P6", "P8"),   ("PO7", "PO5"), ("PO5", "PO3"),
        ("PO3", "POZ"), ("POZ", "PO4"), ("PO4", "PO6"), ("PO6", "PO8"),
        ("CB1", "O1"),  ("O1", "OZ"),   ("OZ", "O2"),   ("O2", "CB2"),
        # Inter-regional longitudinal connections
        ("FP1", "F7"),  ("FP2", "F8"),  ("AF3", "FC3"), ("AF4", "FC4"),
        ("F3", "C3"),   ("F4", "C4"),   ("FZ", "CZ"),   ("C3", "CP3"),
        ("C4", "CP4"),  ("CZ", "CPZ"),  ("CP3", "P3"),  ("CP4", "P4"),
        ("CPZ", "PZ"),  ("P3", "O1"),   ("P4", "O2"),   ("PZ", "OZ"),
    ]

    edges = []
    for c1, c2 in pairs:
        if c1 in ch2idx and c2 in ch2idx:
            i, j = ch2idx[c1], ch2idx[c2]
            edges.append([i, j])
            edges.append([j, i])  # Bidirectional undirected graph

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    torch.save(edge_index, edge_index_path)
    print(f"[Info] Graph constructed: {len(channels)} nodes, {edge_index.shape[1]} directed edges.")
    return edge_index


# ==========================================
# 3. NODE TEMPORAL ENCODER (LSTM)
# ==========================================
class ChannelFeaturizer(nn.Module):
    def __init__(self, in_features=5, hidden_dim=64, num_layers=2, emb_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, emb_dim)
        self.cls = nn.Linear(emb_dim, 3)

    def forward(self, x):
        # x: (B, 1, 5) -> treat 1 as sequence length
        out, _ = self.lstm(x)
        emb = F.relu(self.fc(out[:, -1, :]))
        logits = self.cls(emb)
        return emb, logits


def train_channel_featurizers(X, y, emb_dim=32, device="cpu", save_dir="saved_features"):
    """
    Trains channel-wise temporal feature extractors to generate spatial node representations.
    """
    emb_path = os.path.join(save_dir, f"embeddings_dim{emb_dim}.npy")
    if os.path.exists(emb_path):
        print(f"[Info] Loading cached embeddings from '{emb_path}'...")
        return np.load(emb_path)

    N, num_channels, in_feat = X.shape
    all_embeddings = np.zeros((N, num_channels, emb_dim), dtype=np.float32)

    y_tensor = torch.tensor(y, dtype=torch.long)

    print(f"[Info] Training node featurizers across {num_channels} channels (emb_dim={emb_dim})...")
    for ch in range(num_channels):
        x_ch = torch.tensor(X[:, ch : ch + 1, :], dtype=torch.float32)
        dataset = TensorDataset(x_ch, y_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        model = ChannelFeaturizer(in_features=in_feat, emb_dim=emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(5):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                _, logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Inference for embeddings
        model.eval()
        with torch.no_grad():
            inf_loader = DataLoader(dataset, batch_size=256, shuffle=False)
            ch_embs = []
            for bx, _ in inf_loader:
                bx = bx.to(device)
                emb, _ = model(bx)
                ch_embs.append(emb.cpu().numpy())
            all_embeddings[:, ch, :] = np.concatenate(ch_embs, axis=0)

    np.save(emb_path, all_embeddings)
    print(f"[Info] Embeddings computed and cached: shape={all_embeddings.shape}")
    return all_embeddings


# ==========================================
# 4. GRAPH CLASSIFICATION NETWORK (GCN)
# ==========================================
class GCN_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            # Global mean pooling across node dimension
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)

        return self.classifier(x)


# ==========================================
# 5. TRAINING ROUTINE & EARLY STOPPING
# ==========================================
def train_gcn(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    edge_index,
    device,
    epochs=100,
    patience=15,
    save_path="best_model.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_cnt = 0

    edge_index = edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        running_loss = 0.0

        for i in range(0, X_train.shape[0], 64):
            indices = permutation[i : i + 64]
            bx = X_train[indices].to(device)
            by = y_train[indices].to(device)

            optimizer.zero_grad()
            logits_list = [model(sample, edge_index) for sample in bx]
            logits = torch.cat(logits_list, dim=0)

            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []

        with torch.no_grad():
            for i in range(0, X_val.shape[0], 64):
                bx = X_val[i : i + 64].to(device)
                by = y_val[i : i + 64].to(device)

                logits = torch.cat([model(sample, edge_index) for sample in bx], dim=0)
                v_loss = criterion(logits, by)
                val_losses.append(v_loss.item())

                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.cpu().numpy())

        mean_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(torch.load(save_path))
    return model


def evaluate_model(model, X_test, y_test, edge_index, device):
    model.eval()
    edge_index = edge_index.to(device)
    preds, targets = [], []

    with torch.no_grad():
        for i in range(0, X_test.shape[0], 64):
            bx = X_test[i : i + 64].to(device)
            by = y_test[i : i + 64].to(device)

            logits = torch.cat([model(sample, edge_index) for sample in bx], dim=0)
            p = logits.argmax(dim=-1).cpu().numpy()

            preds.extend(p)
            targets.extend(by.cpu().numpy())

    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, digits=4)
    return acc, report


# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    set_seed(42)
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Step 1: Preprocessing raw SEED data
    X_raw, y_raw = load_eeg_data_corrected(args.data_dir, args.save_dir)

    # Step 2: Build topological electrode graph
    edge_index = build_graph(args.channel_path, args.save_dir)

    # Step 3 & 4: Hyperparameter Grid Search
    emb_dims = [16, 32, 64]
    hidden_dims = [32, 64, 128]
    num_layers_list = [2, 3]

    results = []

    for emb_dim in emb_dims:
        X_emb = train_channel_featurizers(
            X_raw, y_raw, emb_dim=emb_dim, device=device, save_dir=args.save_dir
        )

        # 70% Train, 15% Validation, 15% Test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_emb, y_raw, test_size=0.3, random_state=42, stratify=y_raw
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                run_tag = f"emb{emb_dim}_hid{hidden_dim}_lay{num_layers}"
                print(f"\n--- Running Configuration: {run_tag} ---")

                model = GCN_Classifier(
                    in_channels=emb_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=0.3,
                ).to(device)

                model_save_path = os.path.join(args.save_dir, f"model_{run_tag}.pth")

                model = train_gcn(
                    model,
                    X_train_t,
                    y_train_t,
                    X_val_t,
                    y_val_t,
                    edge_index,
                    device=device,
                    save_path=model_save_path,
                )

                test_acc, test_report = evaluate_model(
                    model, X_test_t, y_test_t, edge_index, device=device
                )
                print(f"[{run_tag}] Test Accuracy: {test_acc:.4f}")
                print(test_report)

                results.append(
                    {
                        "embedding_dim": emb_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "test_accuracy": test_acc,
                    }
                )

    df_results = pd.DataFrame(results)
    results_csv = os.path.join(args.save_dir, "grid_search_results.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\n[Done] All experiments completed. Results saved to '{results_csv}'.")


if __name__ == "__main__":
    main()
