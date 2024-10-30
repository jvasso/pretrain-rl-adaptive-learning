from typing import List

import random
import numpy as np

import networkx as nx
from networkx.drawing.layout import rescale_layout

import matplotlib.pyplot as plt


def generate_random_list(N):
        if N <= 0:
            raise ValueError("N must be a positive integer")
        random_points = [random.random() for _ in range(N - 1)]
        random_points.extend([0, 1])
        random_points.sort()
        random_list = [random_points[i+1] - random_points[i] for i in range(N)]
        return random_list



def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center



def bipartite_layout(
    G, nodes, align="vertical", scale=1, center=None, aspect_ratio=4 / 3
):
    """Position nodes in two straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nodes : list or container
        Nodes in one node set of the bipartite graph.
        This set will be placed on left or top.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    aspect_ratio : number (default=4/3):
        The ratio of the width to the height of the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
    >>> top = nx.bipartite.sets(G)[0]
    >>> pos = nx.bipartite_layout(G, top)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)
    
    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}
    
    height = 1
    width = aspect_ratio * height
    offset = (width / 2, height / 2)

    # moi
    # top = set(nodes)
    # bottom = set(G) - top
    # nodes = list(top) + list(bottom)
    top = nodes
    bottom = [x for x in G if x not in top]
    nodes = top + bottom

    left_xs = np.repeat(0, len(top))
    right_xs = np.repeat(width, len(bottom))
    left_ys = np.linspace(0, height, len(top))
    right_ys = np.linspace(0, height, len(bottom))

    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset
    # moi
    top_pos = np.flip(top_pos, 0)
    bottom_pos = np.flip(bottom_pos, 0)

    pos = np.concatenate([top_pos, bottom_pos])
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos


def grid_layout(nodes_list:List[str], two_dims=True):
    nodes_list = list(nodes_list)
    if two_dims:
        background_nodes = [node for node in nodes_list if "background" in node]
        background_extra_xdim = 0 if len(background_nodes)==0 else 1
        max_xdim = max([int(node.split("_")[0]) for node in nodes_list if "background" not in node]) + 1 + background_extra_xdim
        max_ydim = max([int(node.split("_")[1]) for node in nodes_list]) + 1
        G = nx.grid_2d_graph(max_xdim, max_ydim)
        plt.figure(figsize=(6,6))
        pos = {f"{x}_{y}":(x+background_extra_xdim,-y) for x,y in G.nodes()}
        for background_node in background_nodes:
            pos[background_node] = (0, -int(background_node.split("_")[1]))
        plt.close()
    else:
        max_xdim = max([int(node) for node in nodes_list]) + 1
        max_ydim = 1
        G = nx.grid_2d_graph(max_xdim, max_ydim)
        plt.figure(figsize=(6,6))
        pos = {f"{x}":(x,-y) for x,y in G.nodes()}
        plt.close()
    return pos



def sample_discrete_truncated_exponential(sample_size:int, support_size:int, lam:float):
    """
    Samples multiple numbers from a discretized truncated exponential distribution.
    
    Parameters:
    N (int): The upper limit of the support (1 to N).
    lam (float): The rate parameter of the exponential distribution.
    sample_size (int): The number of samples to draw.

    Returns:
    list: A list of sampled numbers from the distribution.
    """
    # Calculate the probabilities for each value in the support
    probabilities = np.exp(-lam * np.arange(0, support_size ))
    probabilities /= probabilities.sum()
    
    # Sample the specified number of samples based on the calculated probabilities
    return np.random.choice(np.arange(0, support_size), size=sample_size, p=probabilities)