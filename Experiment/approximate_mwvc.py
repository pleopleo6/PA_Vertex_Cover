import networkx as nx
import numpy as np
import pandas as pd
from typing import Set, Tuple, List
import time

def compute_node_weights_from_attenuations(attenuations_df: pd.DataFrame) -> dict:
    attenuations_df['Positive Attenuation'] = -attenuations_df['Attenuation (dB)']
    attenuations_df['Linear Power'] = 10 ** (attenuations_df['Positive Attenuation'] / 10)
    node_weights = {}
    all_nodes = set(attenuations_df['Start Node']).union(set(attenuations_df['End Node']))
    for node in all_nodes:
        relevant_powers = attenuations_df[
            (attenuations_df['Start Node'] == node) | (attenuations_df['End Node'] == node)
        ]['Linear Power']
        node_weights[node] = relevant_powers.sum()
    return node_weights

class ApproximateMWVC:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weights = nx.get_node_attributes(graph, 'weight')
        if not self.weights:
            self.weights = {node: 1.0 for node in graph.nodes()}
    
    def solve(self) -> Tuple[Set[int], float]:
        if len(self.graph.nodes()) < 10:
            return self._solve_small_graph()
        return self._solve_large_graph()

    def _solve_small_graph(self) -> Tuple[Set[int], float]:
        best_cover, best_cost = set(), float('inf')
        for i in range(1, len(self.graph.nodes()) + 1):
            for nodes in self._generate_combinations(list(self.graph.nodes()), i):
                nodes_set = set(nodes)
                if self._is_vertex_cover(nodes_set):
                    cost = sum(self.weights[node] for node in nodes_set)
                    if cost < best_cost:
                        best_cost = cost
                        best_cover = nodes_set
        return best_cover, best_cost

    def _solve_large_graph(self) -> Tuple[Set[int], float]:
        vertex_cover = set()
        total_cost = 0
        G = self.graph.copy()
        edges = list(G.edges())

        while edges:
            best_node = None
            best_score = float('inf')

            for u, v in edges:
                deg_u = G.degree(u)
                deg_v = G.degree(v)
                score_u = self.weights[u] / deg_u if deg_u > 0 else float('inf')
                score_v = self.weights[v] / deg_v if deg_v > 0 else float('inf')

                if score_u < best_score:
                    best_score = score_u
                    best_node = u
                if score_v < best_score:
                    best_score = score_v
                    best_node = v

            vertex_cover.add(best_node)
            total_cost += self.weights[best_node]
            edges = [(x, y) for x, y in edges if best_node not in (x, y)]

        # Post-traitement : tentative de suppression des nœuds inutiles
        for node in list(vertex_cover):
            temp_cover = vertex_cover - {node}
            if self._is_vertex_cover(temp_cover):
                vertex_cover.remove(node)
                total_cost -= self.weights[node]

        return vertex_cover, total_cost

    def _generate_combinations(self, nodes: List[int], k: int) -> List[List[int]]:
        from itertools import combinations
        return list(combinations(nodes, k))

    def _is_vertex_cover(self, selected_nodes: Set[int]) -> bool:
        return all(u in selected_nodes or v in selected_nodes 
                  for u, v in self.graph.edges())

def solve_and_visualize(graph: nx.Graph):
    solver = ApproximateMWVC(graph)
    vertex_cover, cost = solver.solve()
    return vertex_cover, cost

def compare_solutions(exact_cover: Set[int], approx_cover: Set[int], exact_cost: float, approx_cost: float):
    print("\nComparaison des solutions :")
    print(f"Taille exact : {len(exact_cover)}, Coût exact : {exact_cost:.2f}")
    print(f"Taille approximation : {len(approx_cover)}, Coût approximation : {approx_cost:.2f}")
    print(f"Ratio approximation/exact : {approx_cost / exact_cost:.2f}")
    common_nodes = len(exact_cover.intersection(approx_cover))
    total_nodes = len(exact_cover.union(approx_cover))
    similarity = common_nodes / total_nodes if total_nodes > 0 else 0
    print(f"Similarité : {similarity:.2%}")