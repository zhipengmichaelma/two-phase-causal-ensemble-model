# Visualize the causal structures

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import numpy as np


def causality_plotting(strength=None,
                       feature_names=None, 
                       name_size='short',
                       node_size=600, 
                       node_color="yellow", 
                       arrowsize=16, 
                       edge_cmap=plt.cm.Blues, 
                       edge_width=2.5, 
                       label_size=14, 
                       title=None,
                       title_size=16,
                       title_pad=10,
                       margin=0.15,
                       margin_legend=0.7,
                       lay_out="spring_layout",
                       seed=0):
    """
    Visualize the causal structures.

    Parameters
    ----------
    strength: 2d array
        Ensembled causal strength of shape (num_features, num_features)
    feature_names: 1d array
        The array containing name of features of shape (num_features,)
    name_size: str
        "short" or "long". If "short", node names will be attached on nodes.
        If "long", there will be a list of node names attached.
    node_size: float
        The size of nodes
    node_color: str
        The color of nodes
    arrowsize: float
        The size of arrows in edges
    edge_cmap: object
        The colors of the color map
    edge_width: float
        The width of edges
    label_size: float
        The size of labels
    title: str
        Text to use for the title
    title_size: float
        The size of title
    title_pad: float
        The offset of the title from the top of the axes, in points.
    margin: float
        The boundry to arrange the graph position. 
    margin_legend: float
        The boundry to arrange the legend. 
    lay_out:
        The methods of lay_out. "spring_layout" or "circular_layout"
    seed: int
        Set the random state for deterministic node layouts of spring_layout.

    Returns
    -------
    G: object
        the graphical representation of the causal structures.
    """

    feature_names = list(feature_names)
    G = nx.DiGraph()

    # set the nodes
    G.add_nodes_from(range(len(feature_names)))

    # set the edges according to the strength
    edge_list = []
    edge_strength = []
    feature_number = np.arange(0,len(feature_names),1)
    for i in list(itertools.product(feature_number.tolist(),repeat=2)):
        m,n = i
        if strength[m,n] > 0:
            edge_list.append(i)
            edge_strength.append(strength[m,n])
    G.add_edges_from(edge_list)
    
    # positions of nodes
    if lay_out == "spring_layout":
        pos = nx.layout.spring_layout(G,seed=seed)
    if lay_out == "circular_layout":
        pos = nx.layout.circular_layout(G)
    
    # get the boundry of graph
    N = G.number_of_nodes()
    M = G.number_of_edges()
    x_min = 0.
    x_max = 0.
    y_min = 0.
    y_max = 0.
    for i in range(N):
        x_min = np.min([x_min, pos[i][0]])
        x_max = np.max([x_max, pos[i][0]])
        y_min = np.min([y_min, pos[i][1]])
        y_max = np.max([y_max, pos[i][1]])
    
    # set the colors according to strength
    edge_colors = edge_strength
    edge_alphas = edge_strength
    
    # draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    # draw edges
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_size,
        arrowstyle="-|>",
        arrowsize=arrowsize,
        edge_color=edge_colors,
        edge_cmap=edge_cmap,
        edge_vmin=0,
        edge_vmax=1,
        width=edge_width,
        )
    
    # arrange labels
    if name_size == 'short':
        labels = nx.draw_networkx_labels(
            G, 
            pos, 
            labels={n: feature_names[n] for n in G}, 
            font_size=label_size, 
            font_color='k', 
            font_family='Calibri',
            #font_weight='bold', 
        )

    if name_size == 'long':
        labels = nx.draw_networkx_labels(
            G, 
            pos, 
            labels={n: n+1 for n in G}, 
            font_size=label_size, 
            font_color='k', 
            font_family='Calibri',
            #font_weight='bold', 
        )
    
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    
    # set the colorbar
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array([0,1])
    plt.colorbar(pc, fraction=0.04)
    
    # the limits of axis
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(x_min-margin, x_max+margin)
    ax.set_ylim(y_min-margin, y_max+margin)
    
    # arrange the label table if needed
    if name_size == 'long':
        string = ''
        for i in range(N):
            string += f'{i+1}: {feature_names[i]}\n'
        plt.text(
            x = x_max+margin_legend,
            y = y_max+margin,
            s = string,
            ha = 'left',
            va = 'top',
            fontdict = dict(
                fontsize = 14,
                family = 'Calibri',
                weight = 'normal'
                )
        )
    
    # set the title
    plt.title(label=title,
              fontdict={
                  'fontsize': title_size,
                  'fontweight' : 'bold',
                  'verticalalignment': 'baseline',
              },
              loc='center',
              pad=title_pad)
              
    return G
