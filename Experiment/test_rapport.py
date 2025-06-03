import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import deque
from joblib import Parallel, delayed
from min_weighted_vertex_cover import solve_and_visualize as solve_exact
from approximate_mwvc import solve_and_visualize as solve_approx
from fast_approximate_mwvc import solve_and_visualize as solve_fast_approx
from advanced_approximate_mwvc import solve_greedy_matching, solve_local_search

Zc = 50 + 0j  # Impédance caractéristique

def get_random_attenuation_matrix(gamma: complex, length: float, char_impedance: complex) -> np.ndarray:
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag],
                     [base_off_diag / char_impedance, diag]])

def generate_plc_network(num_nodes: int) -> nx.Graph:
    G = nx.random_labeled_rooted_tree(num_nodes)
    G = nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + np.random.uniform(0.01, 0.1) * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, Zc)
    return G

def attenuation_final(matrix):
    cosh_gamma_l = matrix[0, 0]
    return 20 * np.log10(abs(cosh_gamma_l)) - 6.0206

def bfs_and_calculate_attenuation(graph, start_node, index, total_nodes):
    queue = deque([(start_node, np.eye(2))])
    visited = {start_node}
    attenuations = []
    while queue:
        current_node, current_matrix = queue.popleft()
        if current_node != start_node:
            attenuations.append((start_node, current_node, attenuation_final(current_matrix)))
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                new_matrix = np.dot(current_matrix, graph[current_node][neighbor]['ABCD'])
                queue.append((neighbor, new_matrix))
                visited.add(neighbor)
    print(f"Progression: {index}/{total_nodes} nœuds traités.")
    return attenuations

def visualize_graph(G: nx.Graph):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_weight='bold', edge_color='gray', width=1.5)
    plt.title("Generated PLC Network Graph")
    plt.show()

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

def evaluate_algorithms(network, size):
    print(f"\n=== Evaluation of Algorithms on Network with {size} Nodes ===\n")
    results = {}
    
    # Exécuter l'algorithme exact seulement pour les graphes de taille <= 20
    if size <= 20:
        start_time = time.time()
        exact_cover, exact_cost = solve_exact(network)
        exact_time = time.time() - start_time
        results['exact'] = {
            'cover_size': len(exact_cover),
            'cost': exact_cost,
            'time': exact_time
        }
    else:
        results['exact'] = {
            'cover_size': None,
            'cost': None,
            'time': None
        }
    
    # Exécuter l'algorithme d'approximation standard
    start_time = time.time()
    approx_cover, approx_cost = solve_approx(network)
    approx_time = time.time() - start_time
    results['approx'] = {
        'cover_size': len(approx_cover),
        'cost': approx_cost,
        'time': approx_time
    }
    
    # Exécuter l'algorithme d'approximation rapide
    start_time = time.time()
    fast_approx_cover, fast_approx_cost, fast_approx_time = solve_fast_approx(network)
    results['fast_approx'] = {
        'cover_size': len(fast_approx_cover),
        'cost': fast_approx_cost,
        'time': fast_approx_time
    }
    
    # Exécuter l'algorithme Greedy Matching
    start_time = time.time()
    greedy_cover, greedy_cost, greedy_time = solve_greedy_matching(network)
    results['greedy_matching'] = {
        'cover_size': len(greedy_cover),
        'cost': greedy_cost,
        'time': greedy_time
    }
    
    # Exécuter l'algorithme Local Search
    start_time = time.time()
    local_cover, local_cost, local_time = solve_local_search(network)
    results['local_search'] = {
        'cover_size': len(local_cover),
        'cost': local_cost,
        'time': local_time
    }
    
    # Calculer les ratios si l'algorithme exact a été exécuté
    if size <= 20:
        results['approx']['ratio'] = approx_cost / exact_cost if exact_cost > 0 else float('inf')
        results['fast_approx']['ratio'] = fast_approx_cost / exact_cost if exact_cost > 0 else float('inf')
        results['greedy_matching']['ratio'] = greedy_cost / exact_cost if exact_cost > 0 else float('inf')
        results['local_search']['ratio'] = local_cost / exact_cost if exact_cost > 0 else float('inf')
    else:
        results['approx']['ratio'] = None
        results['fast_approx']['ratio'] = None
        results['greedy_matching']['ratio'] = None
        results['local_search']['ratio'] = None
    
    return results

def run_benchmark():
    sizes = [10, 20, 50, 100, 1000, 3000, 5000, 10000]
    results = []
    
    for size in sizes:
        print(f"\n\n--- Testing Graph with {size} Nodes ---")
        network = generate_plc_network(size)
        
        # Compute and set node weights
        node_weights = compute_node_weights_from_graph(network)
        nx.set_node_attributes(network, node_weights, "weight")
        
        # Run evaluation
        eval_results = evaluate_algorithms(network, size)
        
        # Store results
        results.append({
            'Graph Size': size,
            'Exact Cover Size': eval_results['exact']['cover_size'],
            'Exact Cost': eval_results['exact']['cost'],
            'Exact Time': eval_results['exact']['time'],
            'Approx Cover Size': eval_results['approx']['cover_size'],
            'Approx Cost': eval_results['approx']['cost'],
            'Approx Time': eval_results['approx']['time'],
            'Approx Ratio': eval_results['approx']['ratio'],
            'Fast Approx Cover Size': eval_results['fast_approx']['cover_size'],
            'Fast Approx Cost': eval_results['fast_approx']['cost'],
            'Fast Approx Time': eval_results['fast_approx']['time'],
            'Fast Approx Ratio': eval_results['fast_approx']['ratio'],
            'Greedy Matching Cover Size': eval_results['greedy_matching']['cover_size'],
            'Greedy Matching Cost': eval_results['greedy_matching']['cost'],
            'Greedy Matching Time': eval_results['greedy_matching']['time'],
            'Greedy Matching Ratio': eval_results['greedy_matching']['ratio'],
            'Local Search Cover Size': eval_results['local_search']['cover_size'],
            'Local Search Cost': eval_results['local_search']['cost'],
            'Local Search Time': eval_results['local_search']['time'],
            'Local Search Ratio': eval_results['local_search']['ratio']
        })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv('vertex_cover_benchmark_results.csv', index=False)
    print("\nAll results saved to 'vertex_cover_benchmark_results.csv'")
    
    # Plot results
    plot_results(df)

def plot_results(df):
    # Plot execution times
    plt.figure(figsize=(12, 6))
    plt.plot(df['Graph Size'], df['Exact Time'], 'b-o', label='Exact Algorithm')
    plt.plot(df['Graph Size'], df['Approx Time'], 'r-o', label='Standard Approximation')
    plt.plot(df['Graph Size'], df['Fast Approx Time'], 'g-o', label='Fast Approximation')
    plt.plot(df['Graph Size'], df['Greedy Matching Time'], 'y-o', label='Greedy Matching')
    plt.plot(df['Graph Size'], df['Local Search Time'], 'm-o', label='Local Search')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time_comparison.png')
    plt.close()
    
    # Plot solution costs (only for sizes <= 20 where we have exact solutions)
    small_df = df[df['Graph Size'] <= 20]
    plt.figure(figsize=(12, 6))
    plt.plot(small_df['Graph Size'], small_df['Exact Cost'], 'b-o', label='Exact Algorithm')
    plt.plot(small_df['Graph Size'], small_df['Approx Cost'], 'r-o', label='Standard Approximation')
    plt.plot(small_df['Graph Size'], small_df['Fast Approx Cost'], 'g-o', label='Fast Approximation')
    plt.plot(small_df['Graph Size'], small_df['Greedy Matching Cost'], 'y-o', label='Greedy Matching')
    plt.plot(small_df['Graph Size'], small_df['Local Search Cost'], 'm-o', label='Local Search')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Solution Cost')
    plt.title('Solution Cost Comparison (Small Graphs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('solution_cost_comparison.png')
    plt.close()
    
    # Plot approximation ratios
    plt.figure(figsize=(12, 6))
    plt.plot(small_df['Graph Size'], small_df['Approx Ratio'], 'r-o', label='Standard Approximation')
    plt.plot(small_df['Graph Size'], small_df['Fast Approx Ratio'], 'g-o', label='Fast Approximation')
    plt.plot(small_df['Graph Size'], small_df['Greedy Matching Ratio'], 'y-o', label='Greedy Matching')
    plt.plot(small_df['Graph Size'], small_df['Local Search Ratio'], 'm-o', label='Local Search')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Approximation Ratio')
    plt.title('Approximation Ratio Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('approximation_ratio_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_benchmark()