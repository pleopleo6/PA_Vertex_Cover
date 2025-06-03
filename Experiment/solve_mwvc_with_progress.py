import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from BA_att_calc import load_data, create_graph

def load_attenuations():
    return pd.read_csv('attenuations_bfs.csv')

def compute_node_weights_from_attenuations(attenuations_df):
    node_weights = {}
    for _, row in attenuations_df.iterrows():
        start_node = row['Start Node']
        end_node = row['End Node']
        att = row['Attenuation (dB)']
        
        if start_node not in node_weights:
            node_weights[start_node] = 0.0
        if end_node not in node_weights:
            node_weights[end_node] = 0.0
            
        node_weights[start_node] += att
        node_weights[end_node] += att
    return node_weights

def weighted_vertex_cover_greedy(G, weights):
    cover = set()
    edges = list(G.edges())
    progress = []
    
    while edges:
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
        if weights[u] <= weights[v]:
            cover.add(u)
        else:
            cover.add(v)
            
        edges = [(x, y) for x, y in edges if x not in cover and y not in cover]
        
        # Enregistrer la progression
        progress.append({
            'iteration': len(progress),
            'cover_size': len(cover),
            'total_weight': sum(weights[v] for v in cover),
            'remaining_edges': len(edges)
        })
    
    return cover, pd.DataFrame(progress)

def weighted_vertex_cover_refined_greedy(G, weights, max_iter=1000):
    cover, initial_progress = weighted_vertex_cover_greedy(G, weights)
    edges = list(G.edges())
    refinement_progress = []
    
    for i in tqdm(range(max_iter), desc="Phase de raffinement"):
        removed = False
        for node in list(cover):
            cover.remove(node)
            if all(u in cover or v in cover for u, v in edges):
                removed = True
            else:
                cover.add(node)
                
        refinement_progress.append({
            'iteration': i,
            'cover_size': len(cover),
            'total_weight': sum(weights[v] for v in cover)
        })
        
        if not removed:
            break
            
    return cover, initial_progress, pd.DataFrame(refinement_progress)

def plot_progress(initial_progress, refinement_progress):
    # Créer une figure avec 3 sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Taille de la couverture vs Itération
    sns.lineplot(data=initial_progress, x='iteration', y='cover_size', ax=ax1, label='Phase initiale')
    sns.lineplot(data=refinement_progress, x='iteration', y='cover_size', ax=ax1, label='Phase de raffinement')
    ax1.set_title('Évolution de la taille de la couverture')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Taille de la couverture')
    
    # Plot 2: Poids total vs Itération
    sns.lineplot(data=initial_progress, x='iteration', y='total_weight', ax=ax2, label='Phase initiale')
    sns.lineplot(data=refinement_progress, x='iteration', y='total_weight', ax=ax2, label='Phase de raffinement')
    ax2.set_title('Évolution du poids total')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Poids total')
    
    # Plot 3: Arêtes restantes vs Itération
    sns.lineplot(data=initial_progress, x='iteration', y='remaining_edges', ax=ax3)
    ax3.set_title('Évolution du nombre d\'arêtes restantes')
    ax3.set_xlabel('Itération')
    ax3.set_ylabel('Nombre d\'arêtes restantes')
    
    plt.tight_layout()
    plt.savefig('progress_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_with_cover(G, cover, weights):
    plt.figure(figsize=(15, 10))
    
    pos = nx.spring_layout(G, seed=42)
    
    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Dessiner les nœuds
    node_colors = ['red' if node in cover else 'lightblue' for node in G.nodes()]
    node_sizes = [weights[node] * 50 for node in G.nodes()]
    
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
    print("Chargement des données...")
    edges_df, nodes_df, zc_gamma_cenelec_df, zc_gamma_fcc_df = load_data()
    network = create_graph(edges_df, nodes_df, zc_gamma_cenelec_df)
    
    print("Chargement des atténuations...")
    attenuations_df = load_attenuations()
    
    print("Calcul des poids des nœuds...")
    weights = compute_node_weights_from_attenuations(attenuations_df)
    
    print("Résolution du MWVC avec Refined Greedy...")
    cover, initial_progress, refinement_progress = weighted_vertex_cover_refined_greedy(network, weights)
    
    # Afficher les résultats
    total_weight = sum(weights[v] for v in cover)
    print(f"\nRésultats du MWVC:")
    print(f"Taille de la couverture: {len(cover)}")
    print(f"Poids total: {total_weight:.2f}")
    print(f"Pourcentage de nœuds dans la couverture: {(len(cover)/len(network.nodes()))*100:.2f}%")
    
    print("\nGénération des visualisations...")
    plot_progress(initial_progress, refinement_progress)
    plot_network_with_cover(network, cover, weights)
    print("Visualisations sauvegardées dans 'progress_plots.png' et 'network_mwvc.png'")

if __name__ == "__main__":
    main() 