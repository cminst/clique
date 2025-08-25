import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import numpy as np
import argparse
import networkx as nx

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def evaluate_integration_methods(dataset_name='Cora'):
    """Test different ways to integrate the L-RMC component"""

    # Load data
    dataset = Planetoid(root="./data", name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # L-RMC component (convert from 1-indexed Java IDs back to PyG indices)
    lrmc_nodes_java = "1680 2883 901 617 1578 1103"

    # Load the saved remap dictionary from the benchmark script
    import pickle
    remap_filename = f"{dataset_name.lower()}_remap.pkl"
    with open(remap_filename, "rb") as f:
        remap_data = pickle.load(f)
    nodes_sorted = remap_data['nodes_sorted']

    # Convert Java 1-indexed IDs back to PyG indices
    lrmc_nodes_java_ids = [int(x) for x in lrmc_nodes_java.split()]
    lrmc_nodes = [nodes_sorted[j-1] for j in lrmc_nodes_java_ids]

    print(f"Loaded {len(lrmc_nodes)} L-RMC nodes with correct ID mapping")
    print(f"Java IDs: {lrmc_nodes_java_ids[:5]}...")
    print(f"PyG IDs: {lrmc_nodes[:5]}...")

    # Verify connectivity and S(C) on correct nodes
    lrmc_set = set(lrmc_nodes)
    print(f"\nVerification:")
    print(f"L-RMC component size: {len(lrmc_set)}")

    # Check connectivity within the component
    G = nx.Graph()
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Create subgraph of L-RMC component
    lrmc_subgraph = G.subgraph(lrmc_nodes)
    internal_edges = lrmc_subgraph.number_of_edges()
    possible_edges = len(lrmc_nodes) * (len(lrmc_nodes) - 1) // 2
    density = internal_edges / possible_edges if possible_edges > 0 else 0

    print(f"Internal edges in L-RMC: {internal_edges}")
    print(f"Possible edges: {possible_edges}")
    print(f"Density: {density:.4f}")

    # Calculate S(C) - sum of degrees within component
    s_c = 0
    for node in lrmc_nodes:
        neighbors = set(G.neighbors(node))
        internal_neighbors = neighbors & lrmc_set
        s_c += len(internal_neighbors)

    print(f"S(C): {s_c}")
    print(f"Average internal degree: {s_c / len(lrmc_nodes):.2f}")

    integration_methods = {
        "baseline": "no_modification",
        "binary_feature": "add_binary_feature",
        "feature_scaling": "scale_features_2x",
        "feature_scaling_strong": "scale_features_5x",
        "feature_boost": "add_constant_to_features",
        "edge_weights": "boost_internal_edges"
    }

    results = {}

    for method_name, method_type in integration_methods.items():
        print(f"\n{'='*50}")
        print(f"Testing: {method_name}")
        print(f"{'='*50}")

        accuracies = []

        for seed in range(42, 60):  # 5 runs
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Apply integration method
            if method_type == "no_modification":
                x_input = data.x
                edge_index = data.edge_index

            elif method_type == "add_binary_feature":
                component_feature = torch.zeros(data.num_nodes, 1, device=device)
                component_feature[lrmc_nodes] = 1.0
                x_input = torch.cat([data.x, component_feature], dim=1)
                edge_index = data.edge_index

            elif method_type == "scale_features_2x":
                x_input = data.x.clone()
                x_input[lrmc_nodes] *= 2.0
                edge_index = data.edge_index

            elif method_type == "scale_features_5x":
                x_input = data.x.clone()
                x_input[lrmc_nodes] *= 5.0
                edge_index = data.edge_index

            elif method_type == "add_constant_to_features":
                x_input = data.x.clone()
                x_input[lrmc_nodes] += 1.0  # Add constant to all features
                edge_index = data.edge_index

            elif method_type == "boost_internal_edges":
                x_input = data.x
                edge_index = data.edge_index.clone()

                # Create additional "virtual" edges within the component
                lrmc_set = set(lrmc_nodes)
                virtual_edges = []
                for i, u in enumerate(lrmc_nodes):
                    for v in lrmc_nodes[i+1:]:
                        # Add multiple copies of internal edges to boost them
                        for _ in range(3):  # Triple the internal edges
                            virtual_edges.extend([[u, v], [v, u]])

                if virtual_edges:
                    virtual_edge_tensor = torch.tensor(virtual_edges, device=device).t()
                    edge_index = torch.cat([edge_index, virtual_edge_tensor], dim=1)

            # Train model
            model = GCN(x_input.shape[1], 16, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                out = model(x_input, edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred = model(x_input, edge_index).argmax(dim=1)
                        val_correct = pred[data.val_mask] == data.y[data.val_mask]
                        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_correct = pred[data.test_mask] == data.y[data.test_mask]
                        test_acc_at_best_val = int(test_correct.sum()) / int(data.test_mask.sum())

            accuracies.append(test_acc_at_best_val)
            print(f"  Seed {seed}: {test_acc_at_best_val:.4f}")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[method_name] = (mean_acc, std_acc)
        print(f"  Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL INTEGRATION METHOD COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<20} | {'Mean Accuracy':<15} | {'Std Dev':<10} | {'vs Baseline'}")
    print("-" * 70)

    baseline_acc = results["baseline"][0]
    for method, (mean_acc, std_acc) in results.items():
        improvement = mean_acc - baseline_acc
        print(f"{method:<20} | {mean_acc:<15.4f} | {std_acc:<10.4f} | {improvement:+.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate L-RMC integration methods on a specified dataset')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use (default: Cora)')
    args = parser.parse_args()

    evaluate_integration_methods(args.dataset)
