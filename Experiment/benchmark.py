import networkx as nx
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import deque
from joblib import Parallel, delayed
from min_weighted_vertex_cover import solve_and_visualize as solve_exact
from approximate_mwvc import solve_and_visualize as solve_approx

# --- Network Generation Functions ---

def get_random_attenuation_matrix(gamma, length, char_impedance):
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag], 
                     [base_off_diag / char_impedance, diag]])

def generate_plc_network(num_nodes):
    G = nx.random_labeled_rooted_tree(num_nodes)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + 1j * np.random.uniform(0.01, 0.1)
        char_impedance = 50 * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    return G

def attenuation_final(matrix):
    cosh_gamma_l = matrix[0, 0]
    return 20 * np.log10(abs(cosh_gamma_l)) - 6.0206

def bfs_and_calculate_attenuation(graph, start_node):
    queue = deque([(start_node, np.eye(2))])
    visited = {start_node}
    attenuations = []
    while queue:
        current_node, current_matrix = queue.popleft()
        if current_node != start_node:
            final_att = attenuation_final(current_matrix)
            attenuations.append((start_node, current_node, final_att))
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                new_matrix = np.dot(current_matrix, graph[current_node][neighbor]['ABCD'])
                queue.append((neighbor, new_matrix))
                visited.add(neighbor)
    return attenuations

def compute_attenuations(graph):
    total_nodes = len(graph.nodes())
    all_attenuations = Parallel(n_jobs=-1)(
        delayed(bfs_and_calculate_attenuation)(graph, node)
        for node in graph.nodes()
    )
    flat_attenuations = [item for sublist in all_attenuations for item in sublist]
    df = pd.DataFrame(flat_attenuations, columns=['Start Node', 'End Node', 'Attenuation (dB)'])
    df.to_csv('attenuations_plc_network.csv', index=False)
    return df

# --- Experiment Runner ---

def run_experiment(num_nodes):
    graph = generate_plc_network(num_nodes)
    _ = compute_attenuations(graph)

    start_time = time.time()
    exact_cover, exact_cost = solve_exact(graph)
    exact_time = time.time() - start_time

    start_time = time.time()
    approx_cover, approx_cost = solve_approx(graph)
    approx_time = time.time() - start_time

    quality_ratio = approx_cost / exact_cost if exact_cost != 0 else float('inf')

    return {
        'num_nodes': num_nodes,
        'exact_time': exact_time,
        'approx_time': approx_time,
        'exact_cost': exact_cost,
        'approx_cost': approx_cost,
        'quality_ratio': quality_ratio
    }

# --- Plotting Results ---

def plot_results(results):
    sizes = [r['num_nodes'] for r in results]
    exact_times = [r['exact_time'] for r in results]
    approx_times = [r['approx_time'] for r in results]
    quality_ratios = [r['quality_ratio'] for r in results]

    plt.figure()
    plt.plot(sizes, exact_times, marker='o', label='Exact Algorithm')
    plt.plot(sizes, approx_times, marker='o', label='Approximation Algorithm')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Execution Time Comparison')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.figure()
    plt.plot(sizes, quality_ratios, marker='o', color='green')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Approximation / Exact Cost Ratio')
    plt.title('Solution Quality Comparison')
    plt.xscale('log')
    plt.show()

# --- Main ---

def main():
    node_sizes = [10, 25, 50]
    results = [run_experiment(n) for n in node_sizes]
    plot_results(results)

if __name__ == "__main__":
    main()