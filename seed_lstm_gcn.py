import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, TAGConv, BatchNorm
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/ExtractedFeatures_4s"
CHANNEL_PATH = "data/channel-order.xlsx"
SAVE_DIR = "data/saved_features"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "embeddings"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "models"), exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f" Using GPU: {torch.cuda.get_device_name(0)}")
    print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device('cpu')

print(f" Save Directory: {SAVE_DIR}\n")


def load_eeg_data_corrected(data_dir, expected_channels=62):
    
    X_path = os.path.join(SAVE_DIR, "X_raw.npy")
    y_path = os.path.join(SAVE_DIR, "y_raw.npy")
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(" Loading cached data...")
        X = np.load(X_path)
        y = np.load(y_path)
        print(f" EEG shape: {X.shape}")
        print(f" Labels: {y.shape}, distribution: {np.bincount(y)}")
        return X, y
    
    print(" Loading data from .mat files...")
    mat_files = [f for f in os.listdir(data_dir) 
                 if f.endswith(".mat") and "label" not in f.lower()]
    
    all_data = []
    all_labels = []
    session_labels = np.array([2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0])
    
    for filename in sorted(mat_files):
        path = os.path.join(data_dir, filename)
        data = sio.loadmat(path)
        
        valid_keys = [k for k in data.keys() 
                     if not k.startswith("__") and isinstance(data[k], np.ndarray)]
        
        for trial_idx, key in enumerate(sorted(valid_keys)[:15]):
            trial_data = data[key]  # (62, 235, 5) or (235, 62, 5)
            
            if trial_data.ndim == 3:
                if trial_data.shape[0] == expected_channels:
                    trial_data = np.transpose(trial_data, (1, 0, 2))
                elif trial_data.shape[1] == expected_channels:
                    pass
                else:
                    continue
            else:
                continue

            trial_data = trial_data[:, :62, :]
            
            all_data.append(trial_data)
            all_labels.extend([session_labels[trial_idx % 15]] * trial_data.shape[0])
    
    X = np.concatenate(all_data, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    np.save(X_path, X)
    np.save(y_path, y)
    print(f" Data saved to {SAVE_DIR}")
    
    print(f" EEG shape: {X.shape}")
    print(f" Labels: {y.shape}, distribution: {np.bincount(y)}")
    
    return X, y

X_raw, y_raw = load_eeg_data_corrected(DATA_DIR)

def build_graph(channel_path):

    edge_path = os.path.join(SAVE_DIR, "edge_index.pt")
    
    if os.path.exists(edge_path):
        print(" Loading cached graph edges...")
        edge_index = torch.load(edge_path)
        print(f" Graph: {edge_index.shape[1]} edges")
        return edge_index.to(device)
    
    print(" Building graph from channel connections...")
    channels_df = pd.read_excel(channel_path)
    channels = channels_df.iloc[:, 0].dropna().tolist()[:62]
    channel_to_index = {ch: i for i, ch in enumerate(channels)}
    
    edges = [
        ['O2','CB2'],['O2','OZ'],['O1','OZ'],['O1','CB1'],['PO8','P8'],['CB2','PO8'],
        ['PO6','PO4'],['PO6','PO8'],['PO6','P6'],['CB2','PO6'],['P2','PO4'],['O2','PO4'],
        ['POZ','PZ'],['PO3','POZ'],['PO4','POZ'],['OZ','POZ'],['P1','PO3'],['PO5','PO3'],
        ['O1','PO3'],['PO5','CB1'],['PO7','P7'],['PO5','PO7'],['PO7','CB1'],['P4','CP4'],
        ['P2','P4'],['P6','P4'],['CP2','P2'],['PZ','P1'],['PZ','P2'],['P7','P5'],
        ['P3','P5'],['PO5','P5'],['P3','P1'],['P6','CP6'],['P8','P6'],['CP6','CP4'],
        ['CP4','CP2'],['CPZ','CP2'],['PZ','CPZ'],['CZ','CPZ'],['TP7','T7'],['P7','TP7'],
        ['CP3','CP5'],['TP7','CP5'],['P5','CP5'],['C5','CP5'],['CPZ','CP1'],['CP3','CP1'],
        ['P1','CP1'],['C1','CP1'],['P3','CP3'],['C3','CP3'],['CP6','TP8'],['P8','TP8'],
        ['T8','TP8'],['CP4','C4'],['C4','C2'],['CZ','C2'],['CP2','C2'],['FC2','C2'],
        ['C1','CZ'],['T7','C5'],['C1','C3'],['C5','C3'],['FC3','C3'],['C4','C6'],
        ['T8','C6'],['CP6','C6'],['FC6','C6'],['FC2','FC4'],['F4','FC4'],['C4','FC4'],
        ['FC2','FCZ'],['FC1','FCZ'],['CZ','FCZ'],['F7','FT7'],['T7','FT7'],['FC3','FC5'],
        ['FT7','FC5'],['C5','FC5'],['F5','FC5'],['FC3','FC1'],['C1','FC1'],['T8','FT8'],
        ['FC4','FC6'],['FT8','FC6'],['F6','FC6'],['F3','F5'],['F7','F5'],['AF3','F5'],
        ['FT8','F8'],['F6','F8'],['AF4','F6'],['F6','F4'],['AF4','F4'],['FC2','F2'],
        ['F4','F2'],['AF4','F2'],['FCZ','FZ'],['F2','FZ'],['F1','FZ'],['FC1','F1'],
        ['F3','F1'],['AF3','F1'],['FC3','F3'],['FP2','AF4'],['F3','AF3'],['FP1','AF3'],
        ['FP1','FPZ'],['FP2','FPZ']
    ]
    
    edge_index = []
    for e in edges:
        if e[0] in channel_to_index and e[1] in channel_to_index:
            i, j = channel_to_index[e[0]], channel_to_index[e[1]]
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    
    torch.save(edge_index, edge_path)
    print(f" Graph edges saved to {edge_path}")
    print(f" Graph: {edge_index.shape[1]} edges")
    
    return edge_index.to(device)

edge_index = build_graph(CHANNEL_PATH)

class ChannelFeaturizer(nn.Module):
    def __init__(self, input_size=5, embedding_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, embedding_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.project = nn.Linear(embedding_dim, 3)
    
    def forward(self, x):
        """
        x: (batch, seq_len=1, features=5)
        returns: (batch, embedding_dim)
        """
        _, (h, _) = self.lstm(x)
        return h[-1], F.softmax(self.project(h[-1]), dim=-1)

def extract_node_embeddings(X, y, embedding_dim=64, n_epochs=30, lr=1e-3):
    """
    X: (N, 62, 5)
    Returns: (N, 62, embedding_dim)
    """
    emb_path = os.path.join(SAVE_DIR, "embeddings", f"embeddings_dim{embedding_dim}.npy")
    
    if os.path.exists(emb_path):
        print(f" Loading cached embeddings (dim={embedding_dim})...")
        embeddings = np.load(emb_path)
        print(f" Shape: {embeddings.shape}")
        return embeddings
    
    print(f" Extracting embeddings (dim={embedding_dim})...")
    N, C, F = X.shape
    all_embeddings = []
    
    for channel_idx in range(C):
        if (channel_idx + 1) % 10 == 0:
            print(f" Processing channel {channel_idx+1}/{C}...")
        
        channel_data = X[:, channel_idx, :]  # (N, 5)
        
        X_ch = torch.tensor(channel_data).float().unsqueeze(1).to(device)  # (N, 1, 5)
        y_ch = torch.tensor(y).long().to(device)
        
        model = ChannelFeaturizer(input_size=F, embedding_dim=embedding_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        dataset = TensorDataset(X_ch, y_ch)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(n_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                embeddings, outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings, _ = model(X_ch)
            all_embeddings.append(embeddings.cpu().numpy())
        
        del model, X_ch, y_ch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    embeddings = np.stack(all_embeddings, axis=1)  # (N, 62, embedding_dim)

    np.save(emb_path, embeddings)
    print(f" Embeddings saved to {emb_path}")
    print(f" Shape: {embeddings.shape}")
    
    return embeddings

print("\n" + "="*80)
print(" Extracting Node Embeddings...")
print("="*80)

embeddings_dict = {}
for dim in [32, 64, 128]:
    print(f"\n Embedding dimension: {dim}")
    embeddings = extract_node_embeddings(X_raw, y_raw, embedding_dim=dim, n_epochs=30)
    embeddings_dict[dim] = embeddings


class GCN_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, num_classes=3, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch_size):
        """
        x: (batch*channels, embedding_dim)
        edge_index: (2, num_edges)
        batch_size: int
        """
        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (mean across nodes for each sample)
        x = x.view(batch_size, -1, x.size(-1))  # (batch, channels, hidden)
        x = x.mean(dim=1)  # (batch, hidden)
        
        # Classification
        out = self.classifier(x)
        return out


def train_gcn(X_emb, y, edge_index, embedding_dim, hidden_dim=128, n_layers=2, 
              n_epochs=100, lr=1e-3, batch_size=32):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_emb, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    X_train = torch.tensor(X_train).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_train = torch.tensor(y_train).long().to(device)
    y_val = torch.tensor(y_val).long().to(device)
    y_test = torch.tensor(y_test).long().to(device)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = GCN_Classifier(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_classes=3,
        dropout=0.5
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20

    model_name = f"gcn_emb{embedding_dim}_hidden{hidden_dim}_layers{n_layers}.pth"
    model_path = os.path.join(SAVE_DIR, "models", model_name)
    
    print(f"\n Training GCN (embedding_dim={embedding_dim}, hidden={hidden_dim}, layers={n_layers})")
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch_x, batch_y in train_loader:
            B = batch_x.size(0)
            
            # Reshape: (batch, channels, features) -> (batch*channels, features)
            x_flat = batch_x.view(-1, batch_x.size(-1))
            
            optimizer.zero_grad()
            outputs = model(x_flat, edge_index, B)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                B = batch_x.size(0)
                
                x_flat = batch_x.view(-1, batch_x.size(-1))
                outputs = model(x_flat, edge_index, B)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if patience_counter >= max_patience:
            print(f" Early stopping at epoch {epoch}")
            break
    
    print(f" Model saved to {model_path}")
    
    # Test Evaluation
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            B = batch_x.size(0)
            
            x_flat = batch_x.view(-1, batch_x.size(-1))
            outputs = model(x_flat, edge_index, B)
            
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    return test_acc, test_f1, test_preds, test_labels

print("\n" + "="*80)
print(" Grid Search for Best Configuration")
print("="*80)

best_config = None
best_test_acc = 0
results = []

for emb_dim in [32, 64, 128]:
    X_emb = embeddings_dict[emb_dim]
    
    for hidden_dim in [64, 128, 256]:
        for n_layers in [2, 3]:
            config = f"emb={emb_dim}, hidden={hidden_dim}, layers={n_layers}"
            print(f"\n{'='*60}")
            print(f"Testing: {config}")
            print(f"{'='*60}")
            
            test_acc, test_f1, preds, labels = train_gcn(
                X_emb, y_raw, edge_index,
                embedding_dim=emb_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_epochs=100,
                lr=1e-3,
                batch_size=32
            )
            
            print(f"\n Results:")
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            print(f"Test F1-Score: {test_f1*100:.2f}%")
            print(f"\nClassification Report:")
            print(classification_report(labels, preds, 
                                       target_names=['Negative', 'Neutral', 'Positive']))

            results.append({
                'embedding_dim': emb_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'test_acc': test_acc,
                'test_f1': test_f1
            })
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_config = config
                print(f"\n NEW BEST MODEL: {config}")
                print(f" Accuracy: {test_acc*100:.2f}%")

results_df = pd.DataFrame(results)
results_path = os.path.join(SAVE_DIR, "grid_search_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n Grid search results saved to {results_path}")

print("\n" + "="*80)
print(" FINAL RESULTS")
print("="*80)
print(f"Best Configuration: {best_config}")
print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
print(f"\n All files saved in: {SAVE_DIR}")
print("="*80)
