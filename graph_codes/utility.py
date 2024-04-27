import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from graph_from_skeleton.graph_from_skeleton import *
from graph_from_skeleton.utils import *


def add_edge_length(pred_graph): 
    for n1, n2, d in pred_graph.edges(data=True):
    pos1 = np.array(pred_graph.nodes[n1]["pos"])
    pos2 = np.array(pred_graph.nodes[n2]["pos"])
    d["length"] = np.linalg.norm(pos1-pos2)


def calculate_path(graph, source_node):
    return nx.dfs_preorder_nodes(graph.copy(), source_node)


def get_locs(pred_graph, source, locs): 
    for n in nx.dfs_preorder_nodes(pred_graph,source=115):
        locs.append(pred_graph.nodes[n]["pos"])

def cut_loops(pred_graph): 
    return nx.minimum_spanning_tree(pred_graph)

def get_neighbors(graph): 
    neighbors = {}
    for n in graph.nodes:
        neighbors[n] = list(graph.neighbors(n))
    return neighbors

def get_bifurcations(graph, neighbors): 
    bif_nodes = []
    for k,v in neighbors.items():
        if len(v) > 2:
            bif_nodes.append(k)
    return bif_nodes

def remove_edges(graph,threshold, bif_nodes): 
    prune_th = threshold
    pruned_graph = graph.copy()

    for n in bif_nodes:
        T = graph.copy()
        T.remove_node(n)

        subgraphs = list(nx.connected_components(T))
        subgraphs.sort(key=len, reverse=True)

        for sg in subgraphs:
            if len(sg) < prune_th:
                pruned_graph.remove_nodes_from(sg)

def remove_node(graph, node): 
    graph.remove_node(node)
    subgraphs = list(nx.connected_components(graph))
    # to be modified

    
