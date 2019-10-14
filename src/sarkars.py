"""
Ian's python implementation of the combinatorial approach in
Christopher De Sa et al. (2018) Representation Tradeoffs for Hyperbolic Embeddings
"""
import numpy as np
import os
import networkx as nx
import json
import argparse

def isometric_transform(a, x):
    r2 = np.linalg.norm(a)**2 - 1
    return r2/(np.linalg.norm(x - a)**2) * (x-a) + a

def reflect_at_zero(mu, x):
    a = mu/(np.linalg.norm(mu)**2)
    return isometric_transform(a, x)

def add_children(p, x, edge_lengths, verbose=False):

    p0 = reflect_at_zero(x, p)
    x0 = reflect_at_zero(x, x)
    c = len(edge_lengths)
    q = np.linalg.norm(p0)
    p_angle = np.arccos(p0[0]/q)

    if p0[1] < 0:
        p_angle = 2*np.pi - p_angle
    alpha = 2*np.pi/(c+1)

    assert np.linalg.norm(p0) <= 1.0

    points0 = np.zeros((c+1, 2)) # a first row that's empty

    for k in range(c):
        angle = p_angle + alpha*(k+1) # from 1 to c
        points0 [k+1, 0] = edge_lengths[k]*np.cos(angle)
        points0 [k+1, 1] = edge_lengths[k]*np.sin(angle)

    # reflecting them back
    for k in range(c+1):
        points0[k, :] = reflect_at_zero(x, points0[k, :])

    return points0[1:, :]

def hyp_to_euc_dist(x): 
    return np.sqrt((np.cosh(x)-1)/(np.cosh(x)+1)) # hyperb_dist = acosh(1 + 2 \frac{x^2}{1 - x^2}), take inverse
 
def hyp_embedding(tree, root, weighted, tau, keys, verbose=False):
    n = tree.order()
    T = np.zeros((n,2))

    root_children = tree.successors(root)
    d = len(root_children)

    edge_lengths = hyp_to_euc_dist(tau*np.ones(d))


    for i, root_child in enumerate(root_children):
        T[keys[root_child],:]  = edge_lengths[i] * np.array([np.cos(2*np.pi*i/d), np.sin(2*np.pi*i/d)])

    q = []
    q.extend(root_children)

    while len(q)>0:
        h = q[0]
        if verbose:
            print('Placing children of node [{}]'.format(h))

        children = tree.successors(h)
        parent = tree.predecessors(h)[0] 
        num_children = len(children)
        edge_lengths = hyp_to_euc_dist(tau * np.ones(num_children))

        q.extend(children)

        if num_children > 0:
            R = add_children(T[keys[parent], :], T[keys[h], :], edge_lengths, verbose)
            # now we add children
            for child_idx in range(num_children):
                T[keys[children[child_idx]], :] = R[child_idx, :]

        # pop from the the front
        q = q[1:]

    return T



def make_graph(json_path): 
    G = nx.Graph()
    with open(json_path, 'r') as f:
        hierarchy = json.load(f)

    root='root'

    for layer1_class in hierarchy:
        G.add_edge(u=root, v=layer1_class)
        for layer2_subhierarchy in hierarchy[layer1_class]:
            layer2_class = list(layer2_subhierarchy.keys())[0]
            G.add_edge(u=layer1_class, v=layer2_class)
            # subsubhierarchy should be a list
            for layer3_class in layer2_subhierarchy[layer2_class]:
                G.add_edge(u=layer2_class, v=layer3_class)
    return G, root

def run_hyperbolics(json_path, tau, verbose=False):
    G, head = make_graph(json_path)
    G_BFS = nx.bfs_tree(G, head)
    keys = {G_BFS.nodes()[i]:i  for i in range(len(G_BFS.nodes()))}
    embeddings = hyp_embedding(G_BFS, head, False, tau, keys )

    if verbose:
        print(embeddings)
    return embeddings, keys, G_BFS


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Sarkar\'s construction for embedding trees in hyperbolic spaces')
    parser.add_argument('--data', type=str, default='data/hierarchyV1.json',
                        help='path to datafile containing tree')
    parser.add_argument('--tau', type=float, default=0.5, help='')

    args = parser.parse_args()

    # we have a graph G, and we turn it into a tree by BFS
    run_hyperbolics(args.data, args.tau)
    pass