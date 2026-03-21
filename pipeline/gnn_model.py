import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────
def load_graph_and_labels():
    with open("dependency_graph.json") as f:
        graph = json.load(f)
    with open("gnn_training_labels.json") as f:
        labels = json.load(f)
    return graph, labels

# ── Build PyTorch Geometric Data object ───────────────────────────────────────
def build_pyg_data(graph, fault_node_idx):
    # Node features: [in_degree, out_degree, criticality_score,
    #                 total_incoming_calls, is_fault_node]
    node_features = []
    for node in graph["node_features"]:
        features = [
            node["in_degree"],
            node["out_degree"],
            node["criticality_score"],
            node["total_incoming_calls"] / 100.0,  # normalise
            1.0 if node["node_id"] == fault_node_idx else 0.0
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index: [2, num_edges] tensor
    edge_index = torch.tensor(
        [[e["source"] for e in graph["edge_list"]],
         [e["target"] for e in graph["edge_list"]]],
        dtype=torch.long
    )

    # Edge attributes: [weight, avg_duration_ms normalised]
    edge_attr = torch.tensor(
        [[e["weight"], e["avg_duration_ms"] / 1000.0]
         for e in graph["edge_list"]],
        dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ── GNN Model ─────────────────────────────────────────────────────────────────
class BlastRadiusGNN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, edge_dim=2, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=2, edge_dim=2, concat=True)
        self.conv3 = GATConv(hidden_channels * 2, out_channels,
                             heads=1, edge_dim=2, concat=False)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        # Layer 3 — output layer
        x = self.conv3(x, edge_index, edge_attr)
        return torch.sigmoid(x).squeeze(-1)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(graph, labels, epochs=300, lr=0.01):
    model     = BlastRadiusGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("=" * 55)
    print("GNN TRAINING — BlastRadiusGNN")
    print("=" * 55)
    print(f"Architecture: 3-layer GAT")
    print(f"Node features: 5  |  Edge features: 2")
    print(f"Training samples: {len(labels)}")
    print(f"Epochs: {epochs}  |  Learning rate: {lr}")
    print("=" * 55)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for label_data in labels:
            fault_idx = label_data["fault_node_idx"]
            y_true    = torch.tensor(label_data["label_vector"],
                                     dtype=torch.float)
            data      = build_pyg_data(graph, fault_idx)

            optimizer.zero_grad()
            y_pred = model(data)
            loss   = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(labels)
        loss_history.append(avg_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} / {epochs}  |  Loss: {avg_loss:.6f}")

    return model, loss_history

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, graph, labels):
    model.eval()
    service_names = graph["service_names"]

    print("\n" + "=" * 55)
    print("EVALUATION — BLAST RADIUS PREDICTIONS")
    print("=" * 55)

    all_true  = []
    all_pred  = []
    correct_top1 = 0
    correct_top3 = 0

    for label_data in labels:
        fault_idx  = label_data["fault_node_idx"]
        fault_svc  = label_data["fault_service"]
        y_true     = label_data["label_vector"]
        data       = build_pyg_data(graph, fault_idx)

        with torch.no_grad():
            y_pred = model(data).numpy().tolist()

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        # HR@1 — is the fault service the top predicted?
        top1_pred = service_names[np.argmax(y_pred)]
        if top1_pred == fault_svc:
            correct_top1 += 1

        # HR@3 — is the fault service in top 3 predicted?
        top3_idx  = np.argsort(y_pred)[-3:][::-1]
        top3_svcs = [service_names[i] for i in top3_idx]
        if fault_svc in top3_svcs:
            correct_top3 += 1

        print(f"\nFault: {fault_svc}")
        print(f"  {'Service':<16} {'True':>8} {'Predicted':>10} {'Error':>8}")
        print(f"  {'-'*46}")
        for i, svc in enumerate(service_names):
            err    = abs(y_true[i] - y_pred[i])
            marker = " ← FAULT" if svc == fault_svc else ""
            print(f"  {svc:<16} {y_true[i]:>8.3f} {y_pred[i]:>10.3f} {err:>8.3f}{marker}")

    # Overall metrics
    mae = np.mean(np.abs(np.array(all_true) - np.array(all_pred)))
    mse = np.mean((np.array(all_true) - np.array(all_pred))**2)

    print("\n" + "=" * 55)
    print("METRICS SUMMARY")
    print("=" * 55)
    print(f"  MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"  MSE  (Mean Squared Error):   {mse:.4f}")
    print(f"  HR@1 (Top-1 Accuracy):       {correct_top1}/{len(labels)}")
    print(f"  HR@3 (Top-3 Accuracy):       {correct_top3}/{len(labels)}")
    print("=" * 55)

    return mae, mse, loss_history_global

# ── Plot training loss ────────────────────────────────────────────────────────
def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color="#2E75B6", linewidth=2)
    plt.title("GNN Training Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()
    print("training_loss.png saved")

# ── Save model ────────────────────────────────────────────────────────────────
def save_model(model):
    torch.save(model.state_dict(), "blast_radius_gnn.pt")
    print("blast_radius_gnn.pt saved")

# ── Main ──────────────────────────────────────────────────────────────────────
loss_history_global = []

def main():
    global loss_history_global

    graph, labels = load_graph_and_labels()

    # Train
    model, loss_history = train_model(graph, labels, epochs=500, lr=0.005)
    loss_history_global = loss_history

    # Evaluate
    evaluate_model(model, graph, labels)

    # Plot and save
    plot_loss(loss_history)
    save_model(model)

    print("\n" + "=" * 55)
    print("DAY 4 COMPLETE")
    print("=" * 55)
    print("  blast_radius_gnn.pt   — trained GNN model")
    print("  training_loss.png     — loss curve")
    print("=" * 55)

if __name__ == "__main__":
    main()