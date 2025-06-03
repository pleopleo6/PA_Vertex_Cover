import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import deque
from joblib import Parallel, delayed
from min_weighted_vertex_cover import solve_and_visualize as solve_exact
from approximate_mwvc import solve_and_visualize as solve_approx, compare_solutions

# Constants
Zc = 50 + 0j  # Impédance caractéristique

def get_random_attenuation_matrix(gamma: complex, length: float, char_impedance: complex) -> np.ndarray:
    """Computes the ABCD matrix for a two-port network"""
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag], 
                    [base_off_diag / char_impedance, diag]])

def generate_plc_network(num_nodes: int) -> nx.Graph:
    """Generates a PLC network with random parameters and ABCD matrices"""
    # Generate a basic tree structure
    G = nx.random_labeled_rooted_tree(num_nodes)
    
    # Relabel nodes from 0 to num_nodes-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign random parameters and ABCD matrices to each edge
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)  # Small length to minimize attenuation
        gamma = np.random.uniform(0.01, 0.1) + np.random.uniform(0.01, 0.1) * 1j  # Small gamma
        char_impedance = 50 * 1j  # Characteristic impedance
        
        G[u][v]["length"] = length
        G[u][v]["gamma"] = gamma
        G[u][v]["char_impedance"] = char_impedance
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    
    return G

def attenuation_final(matrix):
    """Calculate final attenuation in dB from the ABCD matrix"""
    cosh_gamma_l = matrix[0, 0]
    attenuation_db = 20 * np.log10(abs(cosh_gamma_l)) - 6.0206
    return attenuation_db

def bfs_and_calculate_attenuation(graph, start_node, index, total_nodes):
    """Calculate attenuation using BFS traversal"""
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

    print(f"Progression: {index}/{total_nodes} nœuds traités.")
    return attenuations

def visualize_graph(G: nx.Graph):
    """Visualize the graph using networkx and matplotlib"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, 
            node_color='lightblue', font_weight='bold',
            edge_color='gray', width=1.5)
    plt.title("Generated PLC Network Graph")
    plt.show()

def add_extra_attenuations(attenuations_df: pd.DataFrame) -> pd.DataFrame:
    """Add extra attenuation to specific links to create more interesting cases"""
    # Créer une copie du DataFrame
    modified_df = attenuations_df.copy()
    
    # Ajouter des atténuations supplémentaires sur certains liens
    # Format: (nœud_source, nœud_destination, atténuation_extra)
    extra_attenuations = [
        (0, 1, -8.0),   # Atténuation significative entre les nœuds 0 et 1
        (1, 2, -10.0),  # Atténuation maximale entre les nœuds 1 et 2
        (2, 3, -5.0),   # Atténuation modérée entre les nœuds 2 et 3
    ]
    
    for source, dest, extra_att in extra_attenuations:
        # Trouver l'indice de la ligne correspondante
        mask = ((modified_df['Start Node'] == source) & (modified_df['End Node'] == dest)) | \
               ((modified_df['Start Node'] == dest) & (modified_df['End Node'] == source))
        
        if mask.any():
            # Ajouter l'atténuation supplémentaire
            modified_df.loc[mask, 'Attenuation (dB)'] += extra_att
    
    return modified_df

def main():
    """Main program to generate PLC network, calculate attenuations and visualize"""
    # Generate network
    num_nodes = 5
    network = generate_plc_network(num_nodes)
    
    # Calculate attenuations
    start_time = time.time()
    
    total_nodes = len(network.nodes())
    all_attenuations = Parallel(n_jobs=-1)(
        delayed(bfs_and_calculate_attenuation)(network, node, index, total_nodes)
        for index, node in enumerate(network.nodes(), start=1)
    )
    
    # Flatten attenuation list
    all_attenuations = [item for sublist in all_attenuations for item in sublist]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Temps d'exécution total : {execution_time:.2f} secondes")
    
    # Visualize the network
    visualize_graph(network)
    
    # Save results
    attenuations_df = pd.DataFrame(all_attenuations, 
                                 columns=['Start Node', 'End Node', 'Attenuation (dB)'])
    
    # Ajouter des atténuations supplémentaires
    attenuations_df = add_extra_attenuations(attenuations_df)
    
    attenuations_df.to_csv('attenuations_plc_network.csv', index=False)
    print("Results have been saved to 'attenuations_plc_network.csv'")
    
    # Find and visualize minimum weighted vertex cover using exact algorithm
    print("\nCalcul du vertex cover minimal pondéré (algorithme exact)...")
    exact_cover, exact_cost = solve_exact(network)
    
    # Find and visualize minimum weighted vertex cover using approximation algorithm
    print("\nCalcul du vertex cover minimal pondéré (algorithme d'approximation)...")
    approx_cover, approx_cost = solve_approx(network)
    
    # Compare the solutions
    compare_solutions(exact_cover, approx_cover, exact_cost, approx_cost)

if __name__ == "__main__":
    main() 