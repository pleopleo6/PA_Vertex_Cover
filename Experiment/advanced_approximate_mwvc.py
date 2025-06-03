import networkx as nx
import numpy as np
import time
from typing import Set, Tuple, List
import random

class GreedyMatchingMWVC:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weights = nx.get_node_attributes(graph, 'weight')
        if not self.weights:
            self.weights = {node: 1.0 for node in graph.nodes()}
    
    def solve(self) -> Tuple[Set[int], float, float]:
        start_time = time.time()
        
        # Créer une copie du graphe
        G = self.graph.copy()
        vertex_cover = set()
        total_cost = 0
        
        while G.edges():
            # Trouver l'arête avec le meilleur ratio poids/coût
            best_edge = None
            best_score = float('inf')
            
            for u, v in G.edges():
                # Score basé sur la somme des poids des nœuds
                score = self.weights[u] + self.weights[v]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                # Ajouter les deux nœuds à la couverture
                vertex_cover.add(u)
                vertex_cover.add(v)
                total_cost += self.weights[u] + self.weights[v]
                
                # Supprimer toutes les arêtes incidentes aux deux nœuds
                edges_to_remove = list(G.edges(u)) + list(G.edges(v))
                for edge in edges_to_remove:
                    if G.has_edge(*edge):
                        G.remove_edge(*edge)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return vertex_cover, total_cost, execution_time

class LocalSearchMWVC:
    def __init__(self, graph: nx.Graph, max_iterations: int = 1000):
        self.graph = graph
        self.weights = nx.get_node_attributes(graph, 'weight')
        if not self.weights:
            self.weights = {node: 1.0 for node in graph.nodes()}
        self.max_iterations = max_iterations
    
    def solve(self) -> Tuple[Set[int], float, float]:
        start_time = time.time()
        
        # Initialiser avec une solution gloutonne
        vertex_cover = self._greedy_initial_solution()
        best_cost = sum(self.weights[node] for node in vertex_cover)
        
        # Local Search
        iteration = 0
        while iteration < self.max_iterations:
            # Essayer de supprimer un nœud
            node_to_remove = random.choice(list(vertex_cover))
            temp_cover = vertex_cover - {node_to_remove}
            
            if self._is_vertex_cover(temp_cover):
                new_cost = sum(self.weights[node] for node in temp_cover)
                if new_cost < best_cost:
                    vertex_cover = temp_cover
                    best_cost = new_cost
            
            # Essayer d'échanger deux nœuds
            if len(vertex_cover) >= 2:
                node1, node2 = random.sample(list(vertex_cover), 2)
                temp_cover = vertex_cover - {node1, node2}
                
                # Trouver un nœud qui peut remplacer les deux
                for node in self.graph.nodes():
                    if node not in vertex_cover:
                        test_cover = temp_cover | {node}
                        if self._is_vertex_cover(test_cover):
                            new_cost = sum(self.weights[n] for n in test_cover)
                            if new_cost < best_cost:
                                vertex_cover = test_cover
                                best_cost = new_cost
                                break
            
            iteration += 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return vertex_cover, best_cost, execution_time
    
    def _greedy_initial_solution(self) -> Set[int]:
        G = self.graph.copy()
        vertex_cover = set()
        
        while G.edges():
            best_node = None
            best_score = float('inf')
            
            for node in G.nodes():
                if node not in vertex_cover:
                    score = self.weights[node] / G.degree(node) if G.degree(node) > 0 else float('inf')
                    if score < best_score:
                        best_score = score
                        best_node = node
            
            if best_node is not None:
                vertex_cover.add(best_node)
                edges_to_remove = list(G.edges(best_node))
                for edge in edges_to_remove:
                    G.remove_edge(*edge)
        
        return vertex_cover
    
    def _is_vertex_cover(self, selected_nodes: Set[int]) -> bool:
        return all(u in selected_nodes or v in selected_nodes 
                  for u, v in self.graph.edges())

def solve_greedy_matching(graph: nx.Graph):
    solver = GreedyMatchingMWVC(graph)
    vertex_cover, cost, execution_time = solver.solve()
    print(f"Greedy Matching MWVC:")
    print(f"Cover Size: {len(vertex_cover)}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    return vertex_cover, cost, execution_time

def solve_local_search(graph: nx.Graph):
    solver = LocalSearchMWVC(graph)
    vertex_cover, cost, execution_time = solver.solve()
    print(f"Local Search MWVC:")
    print(f"Cover Size: {len(vertex_cover)}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    return vertex_cover, cost, execution_time 