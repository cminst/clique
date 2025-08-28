import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import argparse

def analyze_lrmc_component(dataset_name='Cora'):
    """Analyze the L-RMC component that was found"""

    # Load dataset (Cora, CiteSeer, or PubMed)
    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0]

    # Convert to NetworkX
    G = nx.Graph()
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # The component found by L-RMC (convert from 1-indexed to 0-indexed)
    lrmc_nodes = "578 644 2437 2631 2633 1676 911 1936 84 469 537 538 3227 1693 2526 2975 2976 803 111 2607 3056 2035 2869 1526 632 1593 2236 3004"

    lrmc_nodes = [int(x) for x in lrmc_nodes.split()]

    print("="*60)
    print("ANALYZING L-RMC COMPONENT")
    print("="*60)

    # Basic stats
    print(f"Component nodes: {lrmc_nodes}")
    print(f"Component size: {len(lrmc_nodes)}")

    # Check if these nodes exist and are connected
    for node in lrmc_nodes:
        if node not in G:
            print(f"ERROR: Node {node} not in graph!")
            return

    # Analyze the induced subgraph
    subgraph = G.subgraph(lrmc_nodes)
    print(f"\nSubgraph analysis:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")
    print(f"  Is connected: {nx.is_connected(subgraph)}")

    # Internal degrees
    internal_degrees = [subgraph.degree(node) for node in lrmc_nodes]
    print(f"  Internal degrees: {internal_degrees}")
    print(f"  Min internal degree: {min(internal_degrees)}")
    print(f"  Max internal degree: {max(internal_degrees)}")
    print(f"  Average internal degree: {np.mean(internal_degrees):.2f}")

    # Compute RMC score manually
    rmc_score = len(lrmc_nodes) * min(internal_degrees)
    print(f"  RMC score (|C| × δ_C): {rmc_score}")

    # Compare to individual node degrees in full graph
    print(f"\nFull graph degrees of component nodes:")
    for node in lrmc_nodes:
        full_degree = G.degree(node)
        internal_degree = subgraph.degree(node)
        print(f"  Node {node}: full degree = {full_degree}, internal degree = {internal_degree}")

    # Find the neighborhoods of these nodes
    print(f"\nNeighborhood analysis:")
    all_neighbors = set()
    for node in lrmc_nodes:
        neighbors = set(G.neighbors(node))
        all_neighbors.update(neighbors)
        print(f"  Node {node} has {len(neighbors)} neighbors in full graph")

    internal_neighbors = all_neighbors.intersection(set(lrmc_nodes))
    external_neighbors = all_neighbors - set(lrmc_nodes)

    print(f"  Internal neighbors: {len(internal_neighbors)}")
    print(f"  External neighbors: {len(external_neighbors)}")

    # Check if this is a clique or near-clique
    max_possible_edges = len(lrmc_nodes) * (len(lrmc_nodes) - 1) // 2
    density = subgraph.number_of_edges() / max_possible_edges
    print(f"  Subgraph density: {density:.3f} (1.0 = complete clique)")

    # Compare to some other candidate subgraphs
    print(f"\n" + "="*60)
    print("COMPARISON TO OTHER SUBGRAPHS")
    print("="*60)

    # Find highest degree nodes
    degrees = dict(G.degree())
    highest_degree_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:5]

    print(f"Top 5 highest degree nodes: {highest_degree_nodes}")
    print(f"Their degrees: {[degrees[n] for n in highest_degree_nodes]}")

    # Analyze their induced subgraph
    high_deg_subgraph = G.subgraph(highest_degree_nodes)
    high_deg_internal = [high_deg_subgraph.degree(node) for node in highest_degree_nodes]
    high_deg_rmc = len(highest_degree_nodes) * min(high_deg_internal)

    print(f"High-degree subgraph:")
    print(f"  Internal degrees: {high_deg_internal}")
    print(f"  RMC score: {high_deg_rmc}")
    print(f"  Is connected: {nx.is_connected(high_deg_subgraph)}")

    # Try a few random 5-node subsets
    print(f"\nRandom 5-node subgraphs (for comparison):")
    np.random.seed(42)
    for i in range(3):
        random_nodes = np.random.choice(list(G.nodes()), 5, replace=False)
        random_subgraph = G.subgraph(random_nodes)
        if random_subgraph.number_of_edges() > 0:
            random_internal = [random_subgraph.degree(node) for node in random_nodes]
            random_rmc = len(random_nodes) * min(random_internal)
            print(f"  Random set {i+1}: RMC = {random_rmc}, min_deg = {min(random_internal)}")

    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"L-RMC found: 5 nodes with RMC score = {rmc_score}")
    print(f"High-degree found: 5 nodes with RMC score = {high_deg_rmc}")

    if rmc_score >= high_deg_rmc:
        print("✓ L-RMC found a better or equal subgraph!")
    else:
        print("✗ L-RMC found a worse subgraph than high-degree selection")

    # Check if the L-RMC component might be a good cluster
    if len(data.y) == len(list(G.nodes())):
        print(f"\nClass labels of L-RMC component:")
        for node in lrmc_nodes:
            if node < len(data.y):
                print(f"  Node {node}: class {data.y[node].item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug L-RMC on a specified dataset')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use (default: Cora)')
    args = parser.parse_args()
    analyze_lrmc_component(args.dataset)
