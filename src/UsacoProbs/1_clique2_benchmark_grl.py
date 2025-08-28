import subprocess
import tempfile
import os
import networkx as nx
from torch_geometric.datasets import Planetoid
import argparse

def debug_lrmc_basic(dataset_name='Cora'):
    """Just get L-RMC working and understand what it's finding"""

    # Load dataset (Cora, CiteSeer, or PubMed)
    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0]

    # Convert to NetworkX for easier analysis
    G = nx.Graph()
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Max degree: {max(dict(G.degree()).values())}")

    # Convert to simple format and save
    # IMPORTANT: Reindex node labels to be contiguous 1..N
    # Planetoid datasets often have non-contiguous node IDs; Java expects 1..N
    filename = f"{dataset_name.lower()}_simple.txt"
    nodes_sorted = sorted(G.nodes())
    remap = {old_id: i + 1 for i, old_id in enumerate(nodes_sorted)}
    
    # Save remap dictionary for use in integration script
    remap_filename = f"{dataset_name.lower()}_remap.pkl"
    import pickle
    with open(remap_filename, "wb") as f:
        pickle.dump({'nodes_sorted': nodes_sorted, 'remap': remap}, f)

    with open(filename, "w") as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        for u, v in G.edges():
            f.write(f"{remap[u]} {remap[v]}\n")

    print(f"\nSaved graph to {filename}")
    print("Try running your Java code manually:")
    print(f"java clique2_mk 1e-4 {filename}")
    print("\nThen try different epsilon values:")
    print(f"java clique2_mk 1e-3 {filename}")
    print(f"java clique2_mk 1e-2 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug L-RMC on a specified dataset')
    parser.add_argument('--dataset', type=str, default='Cora', 
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use (default: Cora)')
    args = parser.parse_args()
    
    debug_lrmc_basic(args.dataset)
