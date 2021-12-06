# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import os
import json
from typing import Optional, Union, Type, List, Dict, IO

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing import nx_pydot
from networkx.drawing.layout import rescale_layout_dict
from networkx.algorithms.tree import is_arborescence
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.traversal import bfs_predecessors
from networkx.readwrite.json_graph import node_link_data, node_link_graph

from deeprob.spn.structure.node import Node, Sum, Product, topological_order
from deeprob.spn.structure.leaf import Leaf, Bernoulli, Categorical, Isotonic, Uniform, Gaussian
from deeprob.spn.structure.cltree import BinaryCLT


def save_digraph_json(graph: nx.DiGraph, f: Union[IO, os.PathLike, str]):
    """
    Save a NetworkX directed graph by using the JSON format.

    :param graph: The NetworkX directed graph.
    :param f: A file-like object or a filepath of the output JSON file.
    """
    # Obtain the JSON object to serialize
    json_obj = json.dumps(node_link_data(graph))

    # Save the object
    if isinstance(f, (os.PathLike, str)):
        with open(f, 'w', encoding='utf-8') as file:
            file.write(json_obj)
    else:
        f.write(json_obj)


def load_digraph_json(f: Union[IO, os.PathLike, str]) -> nx.DiGraph:
    """
    Load a NetworkX directed graph by using the JSON format.

    :param f: A file-like object or a filepath of the input JSON file.
    :return: The NetworkX directed graph.
    """
    # Load the object
    if isinstance(f, (os.PathLike, str)):
        with open(f, 'r', encoding='utf-8') as file:
            json_obj = json.load(file)
    else:
        json_obj = json.load(f)

    # Obtain the NetworkX graph
    graph = node_link_graph(json_obj, directed=True, multigraph=False)
    return graph


def save_spn_json(root: Node, f: Union[IO, os.PathLike, str]):
    """
    Save SPN to file by using the JSON format.

    :param root: The root node of the SPN.
    :param f: A file-like object or a filepath of the output JSON file.
    """
    # Convert the SPN to a NetworkX graph
    graph = spn_to_digraph(root)

    # Save the NetworkX graph
    save_digraph_json(graph, f)


def load_spn_json(f: Union[IO, os.PathLike, str], leaves: Optional[List[Type[Leaf]]] = None) -> Node:
    """
    Load SPN from file by using the JSON format.

    :param f: A file-like object or a filepath of the input JSON file.
    :param leaves: An optional list of custom leaf classes. Useful when dealing with user-defined leaves.
    :return: The loaded SPN with initialied ids for each node.
    :raises ValueError: If multiple custom leaf classes with the same name are defined.
    """
    # Set the default leaf classes map
    leaf_map: Dict[str, Type[Leaf]] = {
        cls.__name__: cls
        for cls in [
            Bernoulli, Categorical, Isotonic, Uniform, Gaussian, BinaryCLT
        ]
    }

    # Augment the leaf mapper dictionary, if custom leaf classes are defined
    if leaves is not None:
        for cls in leaves:
            name = cls.__name__
            if name in leaf_map:
                raise ValueError("Custom leaf class {} already defined".format(name))
            leaf_map[name] = cls

    # Load the NetworkX graph
    graph = load_digraph_json(f)

    # Convert the NetworkX graph to a SPN
    return digraph_to_spn(graph, leaf_map)


def save_binary_clt_json(clt: BinaryCLT, f: Union[IO, os.PathLike, str]):
    """
    Save Binary Chow-Liu Tree (CLT) to file by using the JSON format.

    :param clt: The binary CLT.
    :param f: A file-like object or a filepath of the output JSON file.
    """
    # Convert the CLT to a NetworkX digraph
    graph = binary_clt_to_digraph(clt)

    # Save the NetworkX graph
    save_digraph_json(graph, f)


def load_binary_clt_json(f: Union[IO, os.PathLike, str]) -> BinaryCLT:
    """
    Load Binary Chow-Liu Tree (CLT) from file by using the JSON format.

    :param f: A file-like object or a filepath of the input JSON file.
    :return: The loaded binary CLT.
    """
    # Load the NetworkX graph
    graph = load_digraph_json(f)

    # Convert the NetworkX graph to a binary CLT
    return digraph_to_binary_clt(graph)


def spn_to_digraph(root: Node) -> nx.DiGraph:
    """
    Convert a SPN to a NetworkX directed graph.

    :param root: The root node of the SPN.
    :return: The corresponding NetworkX directed graph.
    :raises ValueError: If the SPN structure is not a directed acyclic graph (DAG).
    """
    # Check the SPN
    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")
    graph = nx.DiGraph()

    # Add nodes to the graph
    for node in nodes:
        if isinstance(node, Sum):
            weights = [round(float(w), 8) for w in node.weights]
            attr = {'class': Sum.__name__, 'scope': node.scope, 'weights': weights}
        elif isinstance(node, Product):
            attr = {'class': Product.__name__, 'scope': node.scope}
        elif isinstance(node, Leaf):
            params = node.params_dict()
            for name, value in params.items():
                if isinstance(value, np.ndarray):  # Convert Numpy arrays into lists
                    if value.dtype in [np.float32, np.float64]:
                        value = value.astype(np.float64)
                        params[name] = np.around(value, 8).tolist()
                    else:
                        params[name] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):  # Convert Numpy floats into Python float
                    params[name] = round(float(value), 8)
                elif isinstance(value, float):  # Round Python floats
                    params[name] = round(value, 8)
            attr = {'class': node.__class__.__name__, 'scope': node.scope, 'params': params}
        else:
            raise ValueError("Unknown node of type {}".format(node.__class__.__name__))
        graph.add_node(node.id, **attr)

    # Add edges to the graph
    for node in nodes:
        for i, c in enumerate(node.children):
            graph.add_edge(c.id, node.id, idx=i)

    return graph


def digraph_to_spn(graph: nx.DiGraph, leaf_map: Dict[str, Type[Leaf]]) -> Node:
    """
    Convert a NetworkX directed graph to a SPN.

    :param graph: The NetworkX directed graph.
    :param leaf_map: The leaf distributions mapper dictionary.
    :return: The corresponding SPN.
    :raises ValueError: If the graph is not a directed acyclic graph (DAG).
    """
    # Check the graph
    if not is_directed_acyclic_graph(graph):
        raise ValueError("The graph is not a directed acyclic graph (DAG)")
    nodes: Dict[int, Leaf] = dict()

    # Instantiate the nodes in the graph
    for node_id in graph.nodes:
        attr = graph.nodes[node_id]
        name = attr['class']
        scope = attr['scope']
        if name == Sum.__name__:
            node = Sum(scope, weights=attr['weights'])
        elif name == Product.__name__:
            node = Product(scope)
        elif name in leaf_map:
            node = leaf_map[name](scope, **attr['params'])
        else:
            raise ValueError("Unknown node of type {}".format(name))
        node.id = node_id
        nodes[node_id] = node

    # Build the edges between the nodes as parent-children dependencies
    for child_id, parent_id in graph.edges:
        idx = graph.edges[child_id, parent_id]['idx']
        parent_node = nodes[parent_id]
        n_children = len(parent_node.children)
        if idx >= n_children:
            parent_node.children.extend([None] * (idx - n_children + 1))
        parent_node.children[idx] = nodes[child_id]

    # Get the root of the SPN
    return nodes[0]


def binary_clt_to_digraph(clt: BinaryCLT) -> nx.DiGraph:
    """
    Convert a binary Chow-Liu Tree (CLT) to a NetworkX directed graph.

    :param clt: The binary CLT.
    :return: The corresponding NetworkX directed graph.
    :raises ValueError: If the CLT is not initialized.
    """
    if clt.tree is None:
        raise ValueError("The CLT's structure must be already initialized")
    graph = nx.DiGraph()

    # Add nodes to the graph
    for node_id in range(len(clt.tree)):
        weight = np.around(clt.params[node_id].astype(np.float64), 8).tolist()
        attr = {'scope': clt.scope[node_id], 'weight': weight}
        graph.add_node(int(node_id), **attr)

    # Add edges to the graph
    for node_id, parent_node_id in enumerate(clt.tree):
        if parent_node_id != -1:
            graph.add_edge(int(parent_node_id), node_id)

    return graph


def digraph_to_binary_clt(graph: nx.DiGraph) -> BinaryCLT:
    """
    Convert a NetworkX directed graph to a binary Chow-Liu Tree (CLT).

    :param graph: The NetworkX directed graph.
    :return: The corresponding Chow-Liu Tree.
    :raises ValueError: If the graph is not a tree.
    """
    # Check the graph and get the root id
    if not is_arborescence(graph):
        raise ValueError("The graph is not a tree")
    root_id = next(node_id for node_id, c in graph.in_degree() if c == 0)

    scope: list = [None] * len(graph)
    tree: list = [None] * len(graph)
    params: list = [None] * len(graph)

    # Include the information about the root node
    attr = graph.nodes[root_id]
    scope[root_id] = attr['scope']
    tree[root_id] = -1
    params[root_id] = attr['weight']

    # Proceed by BFS starting from the root node
    for node_id, parent_id in bfs_predecessors(graph, source=root_id):
        attr = graph.nodes[node_id]
        tree[node_id] = parent_id
        scope[node_id] = attr['scope']
        params[node_id] = attr['weight']

    # Instantiate a Binary CLT
    return BinaryCLT(scope, tree=tree, params=params)


def plot_spn(root: Node, f: Union[IO, os.PathLike, str]):
    """
    Plot a SPN into file.

    :param root: The SPN root node.
    :param f: A file-like object or a filepath of the output file.
    :raises ValueError: If an unknown node type is found.
    :raises ValueError: If the SPN structure is not a DAG.
    """
    # Convert the SPN to a NetworkX directed graph
    graph = spn_to_digraph(root)

    # Build the dictionaries of node labels and colors
    labels = dict()
    colors = dict()
    for node_id in graph.nodes:
        attr = graph.nodes[node_id]
        name = attr['class']
        if name == Sum.__name__:
            label = '+'
            color = '#083d77'
            for child_id, _ in graph.in_edges(node_id):
                idx = graph.edges[child_id, node_id]['idx']
                graph.edges[child_id, node_id]['weight'] = round(attr['weights'][idx], ndigits=2)
        elif name == Product.__name__:
            label = 'x'
            color = '#bf3100'
        else:
            label = repr(attr['scope']).replace(',', '')
            color = '#542188'
        labels[node_id] = label
        colors[node_id] = color

    # Compute the nodes positions using PyDot + Graphviz
    pos = nx_pydot.graphviz_layout(graph, prog='dot')
    pos = {node_id: (x, -y) for node_id, (x, y) in pos.items()}
    pos = rescale_layout_dict(pos)

    # Set the figure size
    figdim = np.maximum(2, np.sqrt(graph.number_of_nodes() + 2 * graph.number_of_edges()))
    plt.figure(figsize=(figdim, figdim))

    # Draw the nodes and edges
    nx.draw_networkx(
        graph, pos=pos, node_color=[colors[node_id] for node_id in graph.nodes],
        labels=labels, arrows=True, font_size=8, font_color='#ffffff'
    )
    nx.draw_networkx_edge_labels(
        graph, pos=pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
        rotate=False, font_size=8, font_color='#000000'
    )

    # Plot the final figure
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.clf()


def plot_binary_clt(clt: BinaryCLT, f: Union[IO, os.PathLike, str], show_weights: bool = True):
    """
    Plot a binary Chow-Liu Tree (CLT) into file.

    :param clt: The binary CLT.
    :param f: A file-like object or a filepath of the output file.
    :param show_weights: Whether to show the conditional probability tables (CPTs).
    """
    # Convert the CLT to a NetworkX directed graph
    graph = binary_clt_to_digraph(clt)

    # Build the dictionary of node labels
    labels = dict()
    for node_id in graph.nodes:
        labels[node_id] = clt.scope[node_id]

    # Compute the nodes positions using PyDot + Graphviz
    pos = nx_pydot.graphviz_layout(graph, prog='dot')
    pos = rescale_layout_dict(pos)

    # Set the figure size
    figdim = np.maximum(2, np.sqrt(graph.number_of_nodes() + 3 * graph.number_of_edges()))
    plt.figure(figsize=(figdim, figdim))

    # Draw the nodes and edges
    nx.draw_networkx(
        graph, pos=pos, node_color='#542188',
        labels=labels, arrows=True, font_size=8, font_color='#ffffff'
    )

    if show_weights:
        # Initialize the edges labels, using the CPTs
        for node_id in graph.nodes:
            attr = graph.nodes[node_id]
            scope = attr['scope']
            weight = attr['weight']
            for child_id, _ in graph.in_edges(node_id):
                cpt = np.around(np.exp(weight), 2)
                label = "$P(X_{{{sc}}}|0)$ {:.2f} {:.2f}\n$P(X_{{{sc}}}|1)$ {:.2f} {:.2f}".format(
                    cpt[0, 0], cpt[0, 1], cpt[1, 0], cpt[1, 1], sc=scope
                )
                graph.edges[child_id, node_id]['weight'] = label

        # Initialize the root node label, using the root CPT
        root_id = next(node_id for node_id, c in graph.in_degree() if c == 0)
        attr = graph.nodes[root_id]
        scope = attr['scope']
        weight = attr['weight']
        cpt = np.around(np.exp(weight), 2)
        root_label = "$P(X_{{{}}})$ {:.2f} {:.2f}".format(scope, cpt[0, 0], cpt[0, 1])

        # Draw root node CPT and other nodes CPTs
        cpt_style_kwargs = {
            'font_size': 6, 'font_color': '#000000', 'font_family': 'monospace',
            'bbox': {'boxstyle': 'round', 'ec': '#444444', 'fc': '#ffffff', 'pad': 0.2}
        }
        root_label_delta = 1.0 + 0.667 / figdim
        nx.draw_networkx_labels(
            graph, pos={root_id: (pos[root_id][0], pos[root_id][1] * root_label_delta)}, labels={root_id: root_label},
            **cpt_style_kwargs
        )
        nx.draw_networkx_edge_labels(
            graph, pos=pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
            rotate=False, **cpt_style_kwargs
        )

    # Plot the final figure
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.clf()
