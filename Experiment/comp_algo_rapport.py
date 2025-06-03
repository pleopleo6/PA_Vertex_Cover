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
    # Generate a random geometric graph
    G = nx.random_geometric_graph(num_nodes, radius=0.3)
    
    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(num_nodes, radius=0.3)
    
    # Add ABCD matrices to edges
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
def local_2approx_vertex_cover(G):
    # Initialisation
    cover = set()
    edges = list(G.edges())
    
    # Calculer les poids des nœuds à partir des matrices ABCD
    weights = compute_node_weights_from_graph(G)
    
    # Calculer les degrés et les poids par arête
    degrees = {node: 0 for node in G.nodes()}
    edge_weights = {}
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
        edge_weights[(u, v)] = min(weights[u], weights[v])
    
    # Premier passage : construire une couverture initiale
    while edges:
        # Trouver l'arête avec le meilleur ratio poids/degre
        best_ratio = float('inf')
        best_edge = None
        best_node = None
        
        for u, v in edges:
            # Calculer le ratio pour chaque nœud de l'arête
            ratio_u = weights[u] / (degrees[u] + 1e-6)
            ratio_v = weights[v] / (degrees[v] + 1e-6)
            
            if ratio_u < best_ratio:
                best_ratio = ratio_u
                best_edge = (u, v)
                best_node = u
            if ratio_v < best_ratio:
                best_ratio = ratio_v
                best_edge = (u, v)
                best_node = v
        
        if best_edge is None:
            break
            
        # Ajouter le meilleur nœud à la couverture
        cover.add(best_node)
        
        # Mettre à jour les degrés
        for u, v in edges[:]:
            if u == best_node or v == best_node:
                other = v if u == best_node else u
                degrees[other] -= 1
                edges.remove((u, v))
    
    # Phase de raffinement
    max_iterations = 500
    for _ in range(max_iterations):
        improved = False
        
        # Essayer de retirer des nœuds de la couverture
        nodes_to_try = sorted(list(cover), key=lambda x: weights[x])
        for node in nodes_to_try:
            # Vérifier si le nœud peut être retiré
            can_remove = True
            affected_edges = []
            uncovered_edges = []
            
            for u, v in G.edges():
                if (u == node or v == node) and u not in cover and v not in cover:
                    can_remove = False
                    uncovered_edges.append((u, v))
                elif (u == node or v == node):
                    affected_edges.append((u, v))
            
            if can_remove and uncovered_edges:
                # Calculer le coût de remplacement
                replacement_weight = 0
                best_replacements = []
                
                # Trier les arêtes non couvertes par poids
                uncovered_edges.sort(key=lambda e: edge_weights[e])
                
                # Essayer différentes combinaisons de remplacement
                for u, v in uncovered_edges:
                    if weights[u] <= weights[v]:
                        best_replacements.append(u)
                        replacement_weight += weights[u]
                    else:
                        best_replacements.append(v)
                        replacement_weight += weights[v]
                
                # Ne remplacer que si le poids total est réduit
                if replacement_weight < weights[node]:
                    cover.remove(node)
                    cover.update(best_replacements)
                    improved = True
                    break
        
        if not improved:
            # Essayer des échanges de nœuds
            for node1 in list(cover):
                for node2 in G.nodes():
                    if node2 not in cover and weights[node2] < weights[node1]:
                        # Vérifier si l'échange est possible
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
        ("Local 2-Approx", local_2approx_vertex_cover)
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