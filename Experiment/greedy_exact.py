import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time as time_module
import seaborn as sns
import pandas as pd
from pulp import *

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

def generate_general_network(num_nodes: int, min_edges_per_node: float = 1.5) -> nx.Graph:
    """Generate a general connected graph with the same attenuation system as PLC networks."""
    G = nx.Graph()
    
    # First create a cycle to ensure connectivity
    for i in range(num_nodes):
        G.add_edge(i, (i + 1) % num_nodes)
    
    # Add additional random edges to increase density
    target_edges = int(num_nodes * min_edges_per_node)
    current_edges = num_nodes  # from the initial cycle
    
    while current_edges < target_edges:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            current_edges += 1
    
    # Add attenuation matrices to all edges
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + 1j * np.random.uniform(0.01, 0.1)
        char_impedance = 50 * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    
    return G

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

def exact_vertex_cover(G, weights):
    # Create the optimization problem
    prob = LpProblem("Vertex_Cover", LpMinimize)
    
    # Create binary variables for each vertex
    x = LpVariable.dicts("vertex", G.nodes(), cat='Binary')
    
    # Objective: minimize the sum of weights of selected vertices
    prob += lpSum([weights[v] * x[v] for v in G.nodes()])
    
    # Constraints: for each edge, at least one of its endpoints must be in the cover
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1
    
    # Strict time limit of 2 seconds
    time_limit = 2
    print(f"Solving for {len(G.nodes())} nodes with time limit of {time_limit} seconds...")
    
    # Solve the problem with a time limit
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)
    
    if status != 1:  # 1 means optimal solution found
        print(f"Warning: Solver did not find optimal solution within {time_limit} seconds")
        print(f"Solver status: {LpStatus[status]}")
        # Return empty set to indicate failure
        return set()
    
    # Get the solution
    cover = {v for v in G.nodes() if value(x[v]) > 0.5}
    return cover

# ---------- Benchmarking ----------
def evaluate_algorithm(G, weights, algo_fn, name):
    start = time_module.time()
    cover = algo_fn(G, weights)
    duration = time_module.time() - start
    
    # Check if we got a valid solution
    if not cover and name == "Exact MWVC":
        return {
            "algorithm": name,
            "cover_size": 0,
            "cover_weight": float('inf'),
            "time": duration
        }
    
    cover_weight = sum(weights[v] for v in cover)
    return {
        "algorithm": name,
        "cover_size": len(cover),
        "cover_weight": cover_weight,
        "time": duration
    }

def benchmark_all_sizes():
    sizes = [10, 50, 100, 1000]  # Testing with larger sizes
    all_results = []

    print("\n=== Testing PLC Networks (Tree Structure) ===")
    for size in sizes:
        print(f"\n--- Benchmarking PLC network with {size} nodes ---")
        G = generate_plc_network(size)
        weights = compute_node_weights_from_graph(G)
        result = evaluate_algorithm(G, weights, exact_vertex_cover, "Exact MWVC (PLC)")
        result["graph_size"] = size
        result["graph_type"] = "PLC"
        all_results.append(result)
        print(f"Nodes: {size:4} | Cover Size: {result['cover_size']:4} | Weight: {result['cover_weight']:.2f} | Time: {result['time']:.4f}s")
        print(f"Number of edges: {G.number_of_edges()}")

    print("\n=== Testing General Networks ===")
    for size in sizes:
        print(f"\n--- Benchmarking general network with {size} nodes ---")
        G = generate_general_network(size)
        weights = compute_node_weights_from_graph(G)
        result = evaluate_algorithm(G, weights, exact_vertex_cover, "Exact MWVC (General)")
        result["graph_size"] = size
        result["graph_type"] = "General"
        all_results.append(result)
        print(f"Nodes: {size:4} | Cover Size: {result['cover_size']:4} | Weight: {result['cover_weight']:.2f} | Time: {result['time']:.4f}s")
        print(f"Number of edges: {G.number_of_edges()}")
    
    return all_results

# ---------- Plotting ----------
def plot_benchmark_results(all_results):
    df = pd.DataFrame(all_results)

    # Plot execution time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="graph_size", y="time", hue="graph_type", marker='o')
    plt.title("Execution Time Comparison: PLC vs Random Networks")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Number of Nodes")
    plt.axhline(y=2, color='r', linestyle='--', label='Time Limit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot cover size
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="graph_size", y="cover_size", hue="graph_type", marker='o')
    plt.title("Cover Size Comparison: PLC vs Random Networks")
    plt.ylabel("Cover Size")
    plt.xlabel("Number of Nodes")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------- Entry Point ----------
if __name__ == "__main__":
    all_results = benchmark_all_sizes()
    plot_benchmark_results(all_results)