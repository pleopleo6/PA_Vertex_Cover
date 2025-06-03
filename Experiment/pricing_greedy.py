import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd

# ---------- Graph Generation ----------
def get_random_attenuation_matrix(gamma: complex, length: float, char_impedance: complex) -> np.ndarray:
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([
        [diag, char_impedance * base_off_diag],
        [base_off_diag / char_impedance, diag]
    ])

def generate_plc_network(num_nodes: int) -> nx.Graph:
    G = nx.random_labeled_rooted_tree(num_nodes)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + 1j * np.random.uniform(0.01, 0.1)
        char_impedance = 50 * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    return G

def compute_node_weights_from_graph(G: nx.Graph) -> dict:
    node_weights = {n: 0.0 for n in G.nodes()}
    for u, v in G.edges():
        att = 1.0  # Default weight if no ABCD matrix
        if 'ABCD' in G[u][v]:
            cosh_gamma_l = G[u][v]['ABCD'][0, 0]
            att = abs(20 * np.log10(abs(cosh_gamma_l)) - 6.0206)
        node_weights[u] += att
        node_weights[v] += att
    return node_weights

# ---------- Vertex Cover Algorithms ----------
def pricing_vertex_cover(G):
    # Initialize prices for edges
    edge_prices = {edge: 0 for edge in G.edges()}
    cover = set()
    uncovered_edges = set(G.edges())
    
    # Get node weights
    weights = compute_node_weights_from_graph(G)
    
    # First phase: Build initial cover using pricing method
    iteration = 0
    max_iterations = len(G.edges()) * 2  # Ensure we don't get stuck in an infinite loop
    
    while uncovered_edges and iteration < max_iterations:
        iteration += 1
        
        # Find edge with minimum price
        min_price = float('inf')
        min_edge = None
        for edge in uncovered_edges:
            if edge_prices[edge] < min_price:
                min_price = edge_prices[edge]
                min_edge = edge
        
        if min_edge is None:
            break
            
        u, v = min_edge
        
        # Calculate the price to pay for each vertex
        price_u = weights[u] - sum(edge_prices[e] for e in G.edges(u) if e in uncovered_edges)
        price_v = weights[v] - sum(edge_prices[e] for e in G.edges(v) if e in uncovered_edges)
        
        # Add the vertex with lower price to cover
        if price_u <= price_v:
            cover.add(u)
            # Update prices for all edges incident to u
            for edge in G.edges(u):
                if edge in uncovered_edges:
                    # Improved pricing strategy: distribute weight based on remaining uncovered edges
                    remaining_edges = len([e for e in G.edges(u) if e in uncovered_edges])
                    if remaining_edges > 0:
                        edge_prices[edge] = weights[u] / remaining_edges
        else:
            cover.add(v)
            # Update prices for all edges incident to v
            for edge in G.edges(v):
                if edge in uncovered_edges:
                    remaining_edges = len([e for e in G.edges(v) if e in uncovered_edges])
                    if remaining_edges > 0:
                        edge_prices[edge] = weights[v] / remaining_edges
        
        # Remove covered edges
        uncovered_edges = {e for e in uncovered_edges if e[0] not in cover and e[1] not in cover}
    
    # Verify that all edges are covered
    if not all(u in cover or v in cover for u, v in G.edges()):
        # If not all edges are covered, use greedy approach as fallback
        remaining_edges = [(u, v) for u, v in G.edges() if u not in cover and v not in cover]
        
        # Sort edges by weight sum for better initial selection
        remaining_edges.sort(key=lambda e: weights[e[0]] + weights[e[1]])
        
        while remaining_edges:
            # Find edge with minimum weight sum
            min_weight = float('inf')
            min_edge = None
            for u, v in remaining_edges:
                weight_sum = weights[u] + weights[v]
                if weight_sum < min_weight:
                    min_weight = weight_sum
                    min_edge = (u, v)
            
            if min_edge is None:
                break
                
            u, v = min_edge
            # Add the vertex with smaller weight
            if weights[u] <= weights[v]:
                cover.add(u)
            else:
                cover.add(v)
                
            # Remove all edges covered by the selected vertex
            remaining_edges = [(x, y) for x, y in remaining_edges if x not in cover and y not in cover]
    
    # Final verification
    if not all(u in cover or v in cover for u, v in G.edges()):
        # If still not all edges are covered, add all remaining uncovered vertices
        for u, v in G.edges():
            if u not in cover and v not in cover:
                if weights[u] <= weights[v]:
                    cover.add(u)
                else:
                    cover.add(v)
    
    # Second phase: Refinement
    max_iterations = 100
    for _ in range(max_iterations):
        improved = False
        
        # Try to remove vertices from cover
        for node in sorted(list(cover), key=lambda x: weights[x]):
            # Check if node can be removed
            can_remove = True
            affected_edges = []
            
            for u, v in G.edges():
                if (u == node or v == node) and u not in cover and v not in cover:
                    can_remove = False
                    break
                elif (u == node or v == node):
                    affected_edges.append((u, v))
            
            if can_remove:
                # Calculate replacement cost
                replacement_cost = 0
                best_replacements = []
                
                for u, v in affected_edges:
                    if u == node:
                        if weights[v] < weights[u]:
                            best_replacements.append(v)
                            replacement_cost += weights[v]
                    else:
                        if weights[u] < weights[v]:
                            best_replacements.append(u)
                            replacement_cost += weights[u]
                
                # Replace only if total weight is reduced
                if replacement_cost < weights[node]:
                    cover.remove(node)
                    cover.update(best_replacements)
                    improved = True
                    break
        
        if not improved:
            # Try vertex swaps
            for node1 in list(cover):
                for node2 in G.nodes():
                    if node2 not in cover and weights[node2] < weights[node1]:
                        # Check if swap is possible
                        can_swap = True
                        for u, v in G.edges():
                            if (u == node1 or v == node1) and u not in cover and v not in cover:
                                if node2 != u and node2 != v:
                                    can_swap = False
                                    break
                        
                        if can_swap:
                            cover.remove(node1)
                            cover.add(node2)
                            improved = True
                            break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return cover

def weighted_vertex_cover_greedy(G, weights):
    cover = set()
    edges = list(G.edges())
    while edges:
        # Find edge with minimum weight sum
        min_weight = float('inf')
        min_edge = None
        for u, v in edges:
            weight_sum = weights[u] + weights[v]
            if weight_sum < min_weight:
                min_weight = weight_sum
                min_edge = (u, v)
        
        if min_edge is None:
            break
            
        u, v = min_edge
        # Add the vertex with smaller weight
        if weights[u] <= weights[v]:
            cover.add(u)
        else:
            cover.add(v)
            
        # Remove all edges covered by the selected vertex
        edges = [(x, y) for x, y in edges if x not in cover and y not in cover]
    return cover

def weighted_vertex_cover_refined_greedy(G, weights, max_iter=1000):
    cover = weighted_vertex_cover_greedy(G, weights)
    edges = list(G.edges())
    
    for _ in range(max_iter):
        removed = False
        for node in list(cover):
            cover.remove(node)
            if all(u in cover or v in cover for u, v in edges):
                removed = True
            else:
                cover.add(node)
        if not removed:
            break
    return cover

# ---------- Benchmarking ----------
def evaluate_algorithm(G, weights, algo_fn, name):
    start = time.time()
    cover = algo_fn(G, weights) if name == "Refined Greedy MWVC" else algo_fn(G)
    duration = time.time() - start
    cover_weight = sum(weights[v] for v in cover)
    return {
        "algorithm": name,
        "cover_size": len(cover),
        "cover_weight": cover_weight,
        "time": duration
    }

def benchmark_all_sizes():
    sizes = [10, 100, 1000]
    algorithms = [
        ("Refined Greedy MWVC", weighted_vertex_cover_refined_greedy),
        ("Pricing Method", pricing_vertex_cover)
    ]
    all_results = []

    for size in sizes:
        print(f"\n--- Benchmarking for {size} nodes ---")
        G = generate_plc_network(size)
        weights = compute_node_weights_from_graph(G)
        for name, algo in algorithms:
            result = evaluate_algorithm(G, weights, algo, name)
            result["graph_size"] = size
            all_results.append(result)
            print(f"{name:25} | Nodes: {size:4} | Cover Size: {result['cover_size']:4} | Weight: {result['cover_weight']:.2f} | Time: {result['time']:.4f}s")
    
    return all_results

# ---------- Plotting ----------
def plot_benchmark_results(all_results):
    df = pd.DataFrame(all_results)

    # Plot cover size
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="graph_size", y="cover_size", hue="algorithm", ax=ax1)
    ax1.set_title("Cover Size vs Graph Size")
    ax1.set_ylabel("Cover Size")
    ax1.set_xlabel("Number of Nodes")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot cover weight
    fig, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="graph_size", y="cover_weight", hue="algorithm", ax=ax2)
    ax2.set_title("Cover Weight vs Graph Size")
    ax2.set_ylabel("Total Weight")
    ax2.set_xlabel("Number of Nodes")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot execution time
    fig, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="graph_size", y="time", hue="algorithm", ax=ax3)
    ax3.set_title("Execution Time vs Graph Size")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_xlabel("Number of Nodes")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# ---------- Entry Point ----------
if __name__ == "__main__":
    all_results = benchmark_all_sizes()
    plot_benchmark_results(all_results)