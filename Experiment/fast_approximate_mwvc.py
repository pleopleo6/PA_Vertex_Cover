import networkx as nx
import numpy as np
import time
from typing import Set, Tuple, List
import heapq

class FastApproximateMWVC:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weights = nx.get_node_attributes(graph, 'weight')
        if not self.weights:
            self.weights = {node: 1.0 for node in graph.nodes()}
    
    def solve(self) -> Tuple[Set[int], float]:
        start_time = time.time()
        
        # Créer une copie du graphe pour ne pas le modifier
        G = self.graph.copy()
        vertex_cover = set()
        total_cost = 0
        
        # Créer une file de priorité pour les arêtes
        edge_heap = []
        for u, v in G.edges():
            # Score basé sur le poids des nœuds et leur degré
            score = min(
                self.weights[u] / G.degree(u),
                self.weights[v] / G.degree(v)
            )
            heapq.heappush(edge_heap, (score, u, v))
        
        # Traiter les arêtes dans l'ordre de priorité
        while edge_heap:
            _, u, v = heapq.heappop(edge_heap)
            
            # Vérifier si l'arête est toujours dans le graphe
            if not G.has_edge(u, v):
                continue
                
            # Choisir le nœud avec le meilleur ratio poids/degre
            if self.weights[u] / G.degree(u) <= self.weights[v] / G.degree(v):
                best_node = u
            else:
                best_node = v
                
            # Ajouter le nœud à la couverture
            vertex_cover.add(best_node)
            total_cost += self.weights[best_node]
            
            # Supprimer toutes les arêtes incidentes au nœud choisi
            edges_to_remove = list(G.edges(best_node))
            for edge in edges_to_remove:
                G.remove_edge(*edge)
                
            # Mettre à jour la file de priorité pour les arêtes restantes
            edge_heap = []
            for u, v in G.edges():
                score = min(
                    self.weights[u] / G.degree(u),
                    self.weights[v] / G.degree(v)
                )
                heapq.heappush(edge_heap, (score, u, v))
        
        # Post-traitement : tentative de suppression des nœuds inutiles
        for node in list(vertex_cover):
            temp_cover = vertex_cover - {node}
            if self._is_vertex_cover(temp_cover):
                vertex_cover.remove(node)
                total_cost -= self.weights[node]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return vertex_cover, total_cost, execution_time
    
    def _is_vertex_cover(self, selected_nodes: Set[int]) -> bool:
        return all(u in selected_nodes or v in selected_nodes 
                  for u, v in self.graph.edges())

def solve_and_visualize(graph: nx.Graph):
    solver = FastApproximateMWVC(graph)
    vertex_cover, cost, execution_time = solver.solve()
    print(f"Fast Approximate MWVC:")
    print(f"Cover Size: {len(vertex_cover)}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    return vertex_cover, cost, execution_time 