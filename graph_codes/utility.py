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


def get_locs_from_graph(graph, source): 
    locs = []
    for n in nx.dfs_preorder_nodes(graph,source=115):
        locs.append(graph.nodes[n]["pos"])
    return locs

def get_locs_and_index(graph, source): 
    locs = []
    for idx, n in enumerate(nx.dfs_preorder_nodes(graph, source=source)):
        locs.append((graph.nodes[n]["pos"], n))
    return locs

def cut_loops(graph): 
    return nx.minimum_spanning_tree(graph)

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

def get_leaf_nodes(graph):
    leaf_nodes = []
    for node in graph.nodes:
        if graph.degree(node) == 1:
            leaf_nodes.append(node)
    return leaf_nodes

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

    
#def return_leaf(): 
    
