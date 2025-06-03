import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BA_att_calc import load_data, create_graph

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

def plot_network_with_cover(G, cover, weights):
    plt.figure(figsize=(15, 10))
    
    # Position des nœuds
    pos = nx.spring_layout(G, seed=42)
    
    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Dessiner les nœuds
    node_colors = ['red' if node in cover else 'lightblue' for node in G.nodes()]
    node_sizes = [weights[node] * 50 for node in G.nodes()]  # Ajuster le facteur selon vos besoins
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Ajouter les labels avec les poids
    labels = {node: f"{node}\n({weights[node]:.1f})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f"Network with MWVC\nCover Size: {len(cover)}, Total Weight: {sum(weights[v] for v in cover):.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('network_mwvc.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Charger les données et créer le graphe
    print("Chargement des données...")
    edges_df, nodes_df, zc_gamma_cenelec_df, zc_gamma_fcc_df = load_data()
    network = create_graph(edges_df, nodes_df, zc_gamma_cenelec_df)
    
    # Calculer les poids des nœuds
    print("Calcul des poids des nœuds...")
    weights = compute_node_weights_from_graph(network)
    
    # Résoudre le MWVC
    print("Résolution du MWVC avec Refined Greedy...")
    cover = weighted_vertex_cover_refined_greedy(network, weights)
    
    # Afficher les résultats
    total_weight = sum(weights[v] for v in cover)
    print(f"\nRésultats du MWVC:")
    print(f"Taille de la couverture: {len(cover)}")
    print(f"Poids total: {total_weight:.2f}")
    print(f"Pourcentage de nœuds dans la couverture: {(len(cover)/len(network.nodes()))*100:.2f}%")
    
    # Visualiser le réseau avec la couverture
    print("\nGénération de la visualisation...")
    plot_network_with_cover(network, cover, weights)
    print("Visualisation sauvegardée dans 'network_mwvc.png'")

if __name__ == "__main__":
    main() 