import numpy as np
import networkx as nx
from collections import deque
import random
import matplotlib.pyplot as plt
from pulp import *
import time

def get_random_attenuation_matrix(gamma, length, char_impedance):
    """Génère une matrice d'atténuation aléatoire ABCD."""
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag], 
                     [base_off_diag / char_impedance, diag]])

def generate_plc_tree(num_nodes):
    """Génère un arbre PLC avec des matrices d'atténuation aléatoires."""
    # Génère un arbre aléatoire avec une racine
    G = nx.random_labeled_rooted_tree(num_nodes)
    
    # Ajoute les matrices d'atténuation sur chaque arête
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + 1j * np.random.uniform(0.01, 0.1)
        char_impedance = 50 * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    return G

def attenuation_final(matrix):
    """Calcule l'atténuation finale à partir de la matrice ABCD."""
    cosh_gamma_l = matrix[0, 0]
    return 20 * np.log10(abs(cosh_gamma_l)) - 6.0206

def calculate_all_attenuations(graph):
    """Calcule toutes les atténuations entre toutes les paires de nœuds."""
    attenuations = []
    for start_node in graph.nodes():
        queue = deque([(start_node, np.eye(2))])
        visited = {start_node}
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

def assign_node_costs(graph, attenuations, threshold=30):
    """Assigne des coûts aux nœuds basés sur les atténuations."""
    node_costs = {node: 0 for node in graph.nodes()}
    
    # Calcul des coûts de base basés sur les atténuations
    for u, v, att in attenuations:
        if att > threshold:  # Si l'atténuation dépasse le seuil
            node_costs[u] += att - threshold
            node_costs[v] += att - threshold
    
    # Ajout d'une variation aléatoire pour chaque nœud
    for node in node_costs:
        # Ajoute une variation aléatoire entre 0.5 et 1.5
        random_factor = np.random.uniform(0.5, 1.5)
        node_costs[node] *= random_factor
    
    # Normalisation des coûts
    max_cost = max(node_costs.values()) if node_costs.values() else 1
    if max_cost == 0:  # Si tous les coûts sont à 0
        for node in node_costs:
            # Attribue un coût aléatoire entre 1 et 10
            node_costs[node] = np.random.uniform(1, 10)
    else:
        for node in node_costs:
            # Normalise entre 1 et 10 avec une variation aléatoire
            base_cost = 1 + (node_costs[node] / max_cost) * 9
            variation = np.random.uniform(-0.5, 0.5)  # Variation de ±0.5
            node_costs[node] = max(1, min(10, base_cost + variation))  # Garde entre 1 et 10
    
    return node_costs

def exact_mvc(G, time_limit=None):
    """Résolution exacte du MVC en utilisant la programmation linéaire."""
    # Créer le problème
    prob = LpProblem("MVC", LpMinimize)
    
    # Variables de décision (1 si le nœud est dans la couverture, 0 sinon)
    x = {node: LpVariable(f'x_{node}', cat='Binary') for node in G.nodes()}
    
    # Fonction objectif (minimiser le nombre de nœuds)
    prob += lpSum(x[node] for node in G.nodes())
    
    # Contraintes
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1  # Au moins un des deux nœuds doit être dans la couverture
    
    # Résoudre le problème avec limite de temps
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    
    # Récupérer la solution
    cover = {node for node in G.nodes() if value(x[node]) > 0.5}
    return cover

def exact_mwvc(G, node_costs, time_limit=None):
    """Résolution exacte du MWVC en utilisant la programmation linéaire."""
    # Créer le problème
    prob = LpProblem("MWVC", LpMinimize)
    
    # Variables de décision (1 si le nœud est dans la couverture, 0 sinon)
    x = {node: LpVariable(f'x_{node}', cat='Binary') for node in G.nodes()}
    
    # Fonction objectif (minimiser la somme des coûts)
    prob += lpSum(node_costs[node] * x[node] for node in G.nodes())
    
    # Contraintes
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1  # Au moins un des deux nœuds doit être dans la couverture
    
    # Résoudre le problème avec limite de temps
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    
    # Récupérer la solution
    cover = {node for node in G.nodes() if value(x[node]) > 0.5}
    return cover

def evaluate_cover(cover, node_costs):
    """Évalue le coût d'une couverture."""
    return sum(node_costs[node] for node in cover)

def visualize_tree(graph, node_costs, cover=None, title="PLC Tree with Costs"):
    """Visualise l'arbre avec les coûts et la couverture."""
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    
    # Dessiner les nœuds
    node_colors = ['red' if cover and node in cover else 'skyblue' for node in graph.nodes()]
    node_sizes = [300 + 50 * node_costs[node] for node in graph.nodes()]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes)
    
    # Dessiner les edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    
    # Ajouter les labels avec les coûts
    labels = {node: f"{node}\n({node_costs[node]:.1f})" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def run_experiment(num_nodes):
    """Exécute une expérience pour une taille d'arbre donnée."""
    # Génération de l'arbre
    G = generate_plc_tree(num_nodes)
    
    # Calcul des atténuations et des coûts
    attenuations = calculate_all_attenuations(G)
    node_costs = assign_node_costs(G, attenuations)
    
    results = {}
    
    # Mesure du temps et des performances pour MVC
    start_time = time.time()
    mvc_cover = exact_mvc(G)
    mvc_time = time.time() - start_time
    
    # Mesure du temps et des performances pour MWVC
    start_time = time.time()
    mwvc_cover = exact_mwvc(G, node_costs)
    mwvc_time = time.time() - start_time
    
    results['MVC'] = {
        'time': mvc_time,
        'cover_size': len(mvc_cover),
        'cover': mvc_cover
    }
    
    results['MWVC'] = {
        'time': mwvc_time,
        'cover_size': len(mwvc_cover),
        'cover': mwvc_cover,
        'total_cost': evaluate_cover(mwvc_cover, node_costs)
    }
    
    return G, node_costs, results

def main():
    # Tailles d'arbres à tester
    sizes = [1000, 5000, 10000]
    
    print("\nAnalyse des performances pour différentes tailles d'arbres")
    print("=" * 60)
    print(f"{'Taille':^10} | {'MVC':^20} | {'MWVC':^20}")
    print(f"{'':^10} | {'Temps (s)':^10} {'Taille':^10} | {'Temps (s)':^10} {'Coût':^10}")
    print("-" * 60)
    
    for size in sizes:
        print(f"\nTest avec {size} nœuds...")
        G, node_costs, results = run_experiment(size)
        
        # Affichage formaté des résultats
        print(f"{size:^10} | {results['MVC']['time']:^10.2f} {results['MVC']['cover_size']:^10} | {results['MWVC']['time']:^10.2f} {results['MWVC']['total_cost']:^10.2f}")
        
        # Affichage détaillé des résultats
        print(f"\nRésultats détaillés pour {size} nœuds:")
        for algo, metrics in results.items():
            print(f"\n{algo}:")
            print(f"  Temps d'exécution: {metrics['time']:.3f} secondes")
            print(f"  Taille de la couverture: {metrics['cover_size']}")
            if algo == 'MWVC':
                print(f"  Coût total: {metrics['total_cost']:.2f}")
                print(f"  Coût moyen par nœud: {metrics['total_cost']/metrics['cover_size']:.2f}")

if __name__ == "__main__":
    main() 