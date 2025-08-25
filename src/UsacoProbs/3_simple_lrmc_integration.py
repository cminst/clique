import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import numpy as np
import argparse

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

    # L-RMC component (convert from 1-indexed)
    lrmc_nodes_from_java = "578 644 2437 2631 2633 1676 911 1936 84 469 537 538 3227 1693 2526 2975 2976 803 111 2607 3056 2035 2869 1526 632 1593 2236 3004"

    lrmc_nodes_from_java = [int(x) for x in lrmc_nodes_from_java.split()]

    nodes_sorted = sorted(data.nodes())
    remap = {old_id: i + 1 for i, old_id in enumerate(nodes_sorted)}

    # given Java id j in 1..N, the original PyG id is nodes_sorted[j-1]
    lrmc_nodes = [nodes_sorted[j-1] for j in lrmc_nodes_from_java]

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
