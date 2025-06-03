import networkx as nx
import numpy as np
import pandas as pd
import time
from collections import deque
from joblib import Parallel, delayed

Zc = 50 + 0j  # Impédance caractéristique

def attenuation_final(matrix):
    cosh_gamma_l = matrix[0, 0]
    attenuation_db = 20 * np.log10(abs(cosh_gamma_l)) - 6.0206
    return attenuation_db

def load_data():
    edges_df = pd.read_excel('edges.xlsx')
    nodes_df = pd.read_excel('nodes.xlsx')

    zc_gamma_cenelec_df = pd.read_csv('zc_gamma_func_cenelec_b.csv', delimiter=';')
    zc_gamma_fcc_df = pd.read_csv('zc_gamma_func_fcc.csv', delimiter=';')

    return edges_df, nodes_df, zc_gamma_cenelec_df, zc_gamma_fcc_df

def create_graph(edges_df, nodes_df, zc_gamma_df):
    G = nx.Graph()

    # Ajouter les nœuds
    G.add_nodes_from(nodes_df['ID_NODE'])

    # Pré-calculer le dictionnaire zc_gamma
    zc_gamma_dict = {row['KEY']: (complex(row['rGAMMA'], row['iGAMMA']), complex(row['rZC'], row['iZC']))
                     for _, row in zc_gamma_df.iterrows()}

    # Ajouter les arêtes avec les matrices ABCD pré-calculées
    for _, row in edges_df.iterrows():
        from_node = row['ID_FROM_NODE']
        to_node = row['ID_TO_NODE']
        length = row['LENGTH']
        key = row['DIAMETER']

        if key in zc_gamma_dict:
            gamma, zc = zc_gamma_dict[key]
            abcd_matrix = calculate_attenuation_matrix(gamma, length, zc)
            G.add_edge(from_node, to_node, ABCD=abcd_matrix, gamma=gamma, length=length, char_impedance=zc)

    return G

def calculate_attenuation_matrix(gamma: complex, length: float, char_impedance: complex) -> np.ndarray:
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag], [base_off_diag / char_impedance, diag]])

def bfs_and_calculate_attenuation(graph, start_node, index, total_nodes):
    queue = deque([(start_node, np.eye(2))])
    visited = {start_node}
    attenuations = []

    while queue:
        current_node, current_matrix = queue.popleft()

        # Calculer l'atténuation seulement si le nœud courant n'est pas le nœud de départ
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

def calculate_path_attenuation(graph, path):
    total_matrix = np.eye(2)
    matrices = {}

    source_matrix = np.array([[1, Zc], [0, 1]])
    total_matrix = np.dot(total_matrix, source_matrix)
    matrices[(path[0], path[0])] = source_matrix

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_abcd = graph[u][v]["ABCD"]
        total_matrix = np.dot(total_matrix, edge_abcd)
        matrices[(u, v)] = edge_abcd

        if len(list(graph.neighbors(u))) > 2 and u != path[0]:
            subtree_impedance = dfs_impedance(graph, u)
            central_matrix = np.array([[1, 0], [1 / subtree_impedance, 1]])
            total_matrix = np.dot(total_matrix, central_matrix)
            matrices[(u, v)] = central_matrix

    return total_matrix, matrices

def dfs_impedance(graph, node):
    visited = set()
    return _dfs_impedance(graph, node, visited)

def _dfs_impedance(graph, node, visited):
    visited.add(node)
    impedances = []

    for neighbor in graph[node]:
        if neighbor not in visited:
            subtree_impedance = _dfs_impedance(graph, neighbor, visited)
            gamma = graph[node][neighbor]["gamma"]
            length = graph[node][neighbor]["length"]
            char_impedance = graph[node][neighbor]["char_impedance"]
            Zbr = subtree_impedance
            Z_eq = calculate_impedance(char_impedance, gamma, length, Zbr)

            if len(list(graph.neighbors(neighbor))) > 1:
                impedances.append(1 / Z_eq)
            else:
                impedances.append(Z_eq)

    if not impedances:
        return Zc

    inv_total_impedance = sum(1 / imp for imp in impedances if imp != 0)
    return 1 / inv_total_impedance if inv_total_impedance != 0 else sum(impedances)

def calculate_impedance(char_impedance, gamma, length, Zbr):
    return char_impedance * (Zbr + char_impedance * np.tanh(gamma * length)) / (
                char_impedance + Zbr * np.tanh(gamma * length))

# Charger les données
edges_df, nodes_df, zc_gamma_cenelec_df, zc_gamma_fcc_df = load_data()

# Créer le graphe en utilisant les données CENELEC B
network = create_graph(edges_df, nodes_df, zc_gamma_cenelec_df)

# Mesurer le temps d'exécution
start_time = time.time()

# Exécuter BFS et calculer l'atténuation pour chaque nœud en parallèle
total_nodes = len(network.nodes())
all_attenuations = Parallel(n_jobs=-1)(
    delayed(bfs_and_calculate_attenuation)(network, node, index, total_nodes)
    for index, node in enumerate(network.nodes(), start=1)
)

# Aplatir la liste des atténuations
all_attenuations = [item for sublist in all_attenuations for item in sublist]

end_time = time.time()
execution_time = end_time - start_time

print(f"Temps d'exécution total : 4854.21 secondes")

# Optionnel : Stocker les atténuations dans un DataFrame pour une analyse ultérieure
attenuations_df = pd.DataFrame(all_attenuations, columns=['Start Node', 'End Node', 'Attenuation (dB)'])
attenuations_df.to_csv('attenuations_bfs.csv', index=False)
