"""Full Coding-Tree Optimizer for Structural Entropy."""

import math
import heapq
import numpy as np

from .incremental import SparseGraph
from .entropy import canonicalize_labels

class CodingTreeNode:
    def __init__(self, id, vol, cut, base_modules, children=None, parent=None, child_h=0, child_cut=0.0):
        self.id = id
        self.vol = vol
        self.cut = cut
        self.base_modules = frozenset(base_modules)
        self.children = set(children) if children is not None else set()
        self.parent = parent
        self.child_h = child_h
        self.child_cut = child_cut
        self.merged = False

def combine_delta(node1: CodingTreeNode, node2: CodingTreeNode, cut_between: float, graph_volume: float) -> float:
    v1 = node1.vol + 1e-12
    v2 = node2.vol + 1e-12
    g1 = node1.cut
    g2 = node2.cut
    v12 = v1 + v2
    
    term1 = (v1 - g1) * math.log2(v12 / v1)
    term2 = (v2 - g2) * math.log2(v12 / v2)
    term3 = -2.0 * cut_between * math.log2(graph_volume / v12)
    
    return (term1 + term2 + term3) / graph_volume

def compress_delta(node: CodingTreeNode, parent: CodingTreeNode) -> float:
    a = node.child_cut
    v1 = node.vol + 1e-12
    v2 = parent.vol + 1e-12
    return a * math.log2(v2 / v1)

def tree_depth(nodes_dict, nid):
    node = nodes_dict[nid]
    depth = 0
    while node.parent is not None:
        node = nodes_dict[node.parent]
        depth += 1
    return depth + nodes_dict[nid].child_h

def compress_node(nodes_dict, node_id, parent_id):
    p_child_h = nodes_dict[parent_id].child_h
    node_children = nodes_dict[node_id].children
    nodes_dict[parent_id].child_cut += nodes_dict[node_id].child_cut
    nodes_dict[parent_id].children.remove(node_id)
    nodes_dict[parent_id].children.update(node_children)
    
    for c in node_children:
        nodes_dict[c].parent = parent_id
        
    com_node_child_h = nodes_dict[node_id].child_h
    nodes_dict.pop(node_id)
    
    if (p_child_h - com_node_child_h) == 1:
        while True:
            if not nodes_dict[parent_id].children:
                max_child_h = 0
            else:
                max_child_h = max([nodes_dict[f_c].child_h for f_c in nodes_dict[parent_id].children])
                
            if nodes_dict[parent_id].child_h == (max_child_h + 1):
                break
            nodes_dict[parent_id].child_h = max_child_h + 1
            parent_id = nodes_dict[parent_id].parent
            if parent_id is None:
                break

def build_coding_tree_from_modules(adj: np.ndarray, base_labels: np.ndarray, target_k: int | None = None):
    base_labels = canonicalize_labels(base_labels)
    n_modules = int(base_labels.max()) + 1 if base_labels.size else 0
    
    graph = SparseGraph.from_adjacency(adj)
    
    module_vol = np.zeros(n_modules, dtype=float)
    module_cut = np.zeros(n_modules, dtype=float)
    
    for i in range(graph.n_nodes):
        cid = base_labels[i]
        module_vol[cid] += graph.degrees[i]
        
    between = np.zeros((n_modules, n_modules), dtype=float)
    for i in range(graph.n_nodes):
        cid1 = base_labels[i]
        for nbr, w in zip(graph.neighbors[i], graph.weights[i]):
            cid2 = base_labels[nbr]
            if cid1 != cid2:
                between[cid1, cid2] += w
                
    for i in range(n_modules):
        module_cut[i] = np.sum(between[i])
        
    nodes = {}
    for i in range(n_modules):
        nodes[i] = CodingTreeNode(i, module_vol[i], module_cut[i], [i])
        
    min_heap = []
    cmp_heap = []
    next_id = n_modules
    
    for i in range(n_modules):
        for j in range(i + 1, n_modules):
            if between[i, j] > 0:
                diff = combine_delta(nodes[i], nodes[j], between[i, j], graph.volume)
                heapq.heappush(min_heap, (diff, i, j, between[i, j]))
                
    unmerged = n_modules
    while unmerged > 1 and min_heap:
        diff, id1, id2, cut_v = heapq.heappop(min_heap)
        if nodes[id1].merged or nodes[id2].merged:
            continue
            
        nodes[id1].merged = True
        nodes[id2].merged = True
        
        v = nodes[id1].vol + nodes[id2].vol
        g = nodes[id1].cut + nodes[id2].cut - 2 * cut_v
        child_h = max(nodes[id1].child_h, nodes[id2].child_h) + 1
        
        new_node = CodingTreeNode(next_id, v, g, nodes[id1].base_modules | nodes[id2].base_modules, 
                                  children=[id1, id2], child_h=child_h, child_cut=cut_v)
        nodes[id1].parent = next_id
        nodes[id2].parent = next_id
        nodes[next_id] = new_node
        
        # Merge adjacency
        between_new = np.zeros((next_id + 1, next_id + 1))
        between_new[:between.shape[0], :between.shape[1]] = between
        between = between_new
        
        for i in range(next_id):
            if i in nodes and not nodes[i].merged:
                w = between[id1, i] + between[id2, i]
                if w > 0:
                    between[next_id, i] = w
                    between[i, next_id] = w
                    
                    new_diff = combine_delta(nodes[i], nodes[next_id], w, graph.volume)
                    heapq.heappush(min_heap, (new_diff, i, next_id, w))
                    
        if nodes[id1].child_h > 0:
            heapq.heappush(cmp_heap, [compress_delta(nodes[id1], nodes[next_id]), id1, next_id])
        if nodes[id2].child_h > 0:
            heapq.heappush(cmp_heap, [compress_delta(nodes[id2], nodes[next_id]), id2, next_id])
            
        next_id += 1
        unmerged -= 1
        
    root = next_id - 1
    
    if unmerged > 1:
        unmerged_nodes = {i for i, n in nodes.items() if not n.merged}
        new_child_h = max([nodes[i].child_h for i in unmerged_nodes]) + 1
        new_node = CodingTreeNode(next_id, graph.volume, 0, set(range(n_modules)), 
                                  children=unmerged_nodes, child_h=new_child_h)
        nodes[next_id] = new_node
        for i in unmerged_nodes:
            nodes[i].merged = True
            nodes[i].parent = next_id
            if nodes[i].child_h > 0:
                heapq.heappush(cmp_heap, [compress_delta(nodes[i], nodes[next_id]), i, next_id])
        root = next_id

    # Compress Phase
    if target_k is not None:
        while nodes[root].child_h > target_k and cmp_heap:
            diff, node_id, p_id = heapq.heappop(cmp_heap)
            if tree_depth(nodes, node_id) <= target_k:
                continue
            children = list(nodes[node_id].children)
            compress_node(nodes, node_id, p_id)
            
            if nodes[root].child_h == target_k:
                break
                
            for e in cmp_heap:
                if e[1] == p_id:
                    if tree_depth(nodes, p_id) > target_k:
                        e[0] = compress_delta(nodes[e[1]], nodes[e[2]])
                if e[1] in children:
                    if nodes[e[1]].child_h == 0:
                        continue
                    if tree_depth(nodes, e[1]) > target_k:
                        e[2] = p_id
                        e[0] = compress_delta(nodes[e[1]], nodes[p_id])
            heapq.heapify(cmp_heap)
            
    return root, nodes

def extract_flat_labels(nodes, root, n_nodes, base_labels):
    labels = np.zeros(n_nodes, dtype=np.int32)
    for i, child_id in enumerate(nodes[root].children):
        for mod in nodes[child_id].base_modules:
            labels[base_labels == mod] = i
    return canonicalize_labels(labels)
