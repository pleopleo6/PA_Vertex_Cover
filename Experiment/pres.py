import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from itertools import combinations

# ==============================================
# 1. Génération des graphes
# ==============================================

def generate_random_graph(n, p=0.3):
    """Génère un graphe aléatoire connexe (graphe général)"""
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    return G

def generate_random_tree(n):
    """Génère un arbre aléatoire"""
    return nx.random_labeled_rooted_tree(n)

# ==============================================
# 2. Algorithmes pour MVC/MWVC
# ==============================================

def is_vertex_cover(graph, cover):
    """Vérifie si un ensemble de sommets est une couverture"""
    for u, v in graph.edges():
        if u not in cover and v not in cover:
            return False
    return True

def exact_mvc(graph):
    """Algorithme exact (force brute) pour MVC"""
    n = len(graph)
    min_cover = set(range(n))
    
    # Test toutes les tailles possibles de couverture
    for k in range(1, n + 1):
        # Test toutes les combinaisons possibles de k sommets
        for cover in combinations(range(n), k):
            if is_vertex_cover(graph, set(cover)):
                return set(cover)
    
    return min_cover

def dynamic_mvc_tree(tree):
    """Algorithme DP optimal O(n) pour MVC dans un arbre"""
    dp = {node: [0, 1] for node in tree.nodes()}  # [non pris, pris]
    
    # Parcours post-ordre
    for u in nx.dfs_postorder_nodes(tree):
        for v in tree.neighbors(u):
            if v in dp:  # Pour éviter les doublons (car arbre non orienté)
                dp[u][0] += dp[v][1]           # Si u non pris, v doit être pris
                dp[u][1] += min(dp[v][0], dp[v][1])  # Si u pris, v peut être pris ou non
    
    # Reconstruction de la solution
    cover = set()
    root = next(iter(tree.nodes()))
    stack = [(root, None, dp[root][1] < dp[root][0])]
    
    while stack:
        u, parent, taken = stack.pop()
        if taken:
            cover.add(u)
        for v in tree.neighbors(u):
            if v != parent:
                if taken:
                    stack.append((v, u, dp[v][1] < dp[v][0]))
                else:
                    stack.append((v, u, True))
    
    return cover

# ==============================================
# 3. Expériences et visualisation
# ==============================================

def run_experiments(max_n=20, step=1):
    """Compare les temps d'exécution sur graphes généraux vs arbres"""
    sizes = range(5, max_n, step)
    times_general = []
    times_tree = []
    theoretical_general = []
    theoretical_tree = []
    
    for n in tqdm(sizes):
        # Graphe général
        G = generate_random_graph(n)
        start = time.time()
        exact_mvc(G)
        times_general.append(time.time() - start)
        theoretical_general.append(2**n / 1e8)  # Normalisation ajustée pour rester sous la courbe expérimentale
        
        # Arbre
        T = generate_random_tree(n)
        start = time.time()
        dynamic_mvc_tree(T)
        times_tree.append(time.time() - start)
        theoretical_tree.append(n / 1e6)  # Normalisation pour les arbres
    
    return sizes, times_general, times_tree, theoretical_general, theoretical_tree

def plot_results(sizes, times_general, times_tree, theoretical_general, theoretical_tree):
    """Visualise les résultats de complexité"""
    plt.figure(figsize=(15, 5))
    
    # Graphique logarithmique pour les graphes généraux
    plt.subplot(1, 2, 1)
    plt.semilogy(sizes, times_general, 'r-', label="Graphe général (Exact)")
    plt.semilogy(sizes, theoretical_general, 'r--', label="O(2^n) théorique")
    plt.xlabel("Taille du graphe")
    plt.ylabel("Temps (s) - échelle log")
    plt.title("Complexité dans les graphes généraux")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Graphique logarithmique pour les arbres
    plt.subplot(1, 2, 2)
    plt.semilogy(sizes, times_tree, 'b-', label="Arbre (DP)")
    plt.semilogy(sizes, theoretical_tree, 'b--', label="O(n) théorique")
    plt.xlabel("Taille de l'arbre")
    plt.ylabel("Temps (s) - échelle log")
    plt.title("Complexité dans les arbres")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==============================================
# 4. Illustration graphique
# ==============================================

def visualize_example():
    """Montre un exemple de solution sur petit graphe"""
    G = generate_random_graph(10)
    T = generate_random_tree(10)
    
    mvc_general = exact_mvc(G)
    mvc_tree = dynamic_mvc_tree(T)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=['red' if n in mvc_general else 'skyblue' for n in G.nodes()])
    plt.title("MVC exact dans un graphe général")
    
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(T)
    nx.draw(T, pos, with_labels=True, node_color=['red' if n in mvc_tree else 'skyblue' for n in T.nodes()])
    plt.title("MVC optimal dans un arbre (DP)")
    
    plt.tight_layout()
    plt.show()

# ==============================================
# Exécution principale
# ==============================================

if __name__ == "__main__":
    print("Comparaison de complexité MVC...")
    sizes, times_general, times_tree, theoretical_general, theoretical_tree = run_experiments(max_n=20, step=1)
    plot_results(sizes, times_general, times_tree, theoretical_general, theoretical_tree)
    
    print("\nExemple visuel sur petits graphes:")
    visualize_example()