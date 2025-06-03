import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Set, Tuple, List
import time
from itertools import chain, combinations

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

class FullExhaustiveMWVC:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.attenuations_df = pd.read_csv('attenuations_plc_network.csv')
        self.weights = compute_node_weights_from_attenuations(self.attenuations_df)

    def _is_vertex_cover(self, selected_nodes: Set[int]) -> bool:
        return all(u in selected_nodes or v in selected_nodes for u, v in self.graph.edges())

    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    def solve(self) -> Tuple[Set[int], float]:
        best_cover = set()
        best_cost = float('inf')

        start_time = time.time()
        for candidate in self._powerset(self.nodes):
            candidate_set = set(candidate)
            if self._is_vertex_cover(candidate_set):
                cost = sum(self.weights.get(n, 1.0) for n in candidate_set)
                if cost < best_cost:
                    best_cover = candidate_set
                    best_cost = cost
        end_time = time.time()

        print(f"Temps de résolution exhaustive : {end_time - start_time:.2f} secondes")
        return best_cover, best_cost

def visualize_vertex_cover(graph: nx.Graph, vertex_cover: Set[int], title: str = "Optimal Minimum Weighted Vertex Cover"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=700, nodelist=[n for n in graph.nodes() if n not in vertex_cover])
    nx.draw_networkx_nodes(graph, pos, node_color='red', node_size=700, nodelist=list(vertex_cover))
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=1.5)
    plt.title(title)
    plt.show()

def solve_and_visualize(graph: nx.Graph):
    solver = FullExhaustiveMWVC(graph)
    vertex_cover, cost = solver.solve()
    print(f"Taille du vertex cover (exhaustive) : {len(vertex_cover)}")
    print(f"Coût total (exhaustive) : {cost:.2f}")
    print(f"Nœuds dans le vertex cover (exhaustive) : {sorted(list(vertex_cover))}")
    visualize_vertex_cover(graph, vertex_cover)
    return vertex_cover, cost