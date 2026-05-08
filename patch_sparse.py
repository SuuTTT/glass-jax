import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

# 1. Update SparseGraph dataclass
old_dataclass = """class SparseGraph:
    \"\"\"Neighbor-list view of a non-negative undirected graph.\"\"\"

    neighbors: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    degrees: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int"""
new_dataclass = """class SparseGraph:
    \"\"\"Neighbor-list view of a non-negative undirected graph.\"\"\"

    neighbors: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    degrees: np.ndarray
    node_cuts: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int"""
content = content.replace(old_dataclass, new_dataclass)

# 2. Update from_adjacency
old_from_adj = """    @classmethod
    def from_adjacency(cls, adj: np.ndarray) -> "SparseGraph":
        matrix = as_symmetric_adjacency(adj)
        neighbors = []
        weights = []
        degrees = matrix.sum(axis=1)
        edge_count = 0
        for node in range(matrix.shape[0]):
            idx = np.flatnonzero(matrix[node] > 0)
            vals = matrix[node, idx].astype(float, copy=True)
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals)
            edge_count += int(np.sum(idx > node))
        return cls(
            neighbors=tuple(neighbors),
            weights=tuple(weights),
            degrees=degrees.astype(float, copy=False),
            volume=float(degrees.sum()),
            n_nodes=int(matrix.shape[0]),
            n_edges=edge_count,
        )"""
new_from_adj = """    @classmethod
    def from_adjacency(cls, adj: np.ndarray) -> "SparseGraph":
        matrix = np.asarray(adj, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        degrees = matrix.sum(axis=1)
        np.fill_diagonal(matrix, 0.0)
        node_cuts = matrix.sum(axis=1)
        
        neighbors = []
        weights = []
        edge_count = 0
        for node in range(matrix.shape[0]):
            idx = np.flatnonzero(matrix[node] > 0)
            vals = matrix[node, idx].astype(float, copy=True)
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals)
            edge_count += int(np.sum(idx > node))
        return cls(
            neighbors=tuple(neighbors),
            weights=tuple(weights),
            degrees=degrees.astype(float, copy=False),
            node_cuts=node_cuts.astype(float, copy=False),
            volume=float(degrees.sum()),
            n_nodes=int(matrix.shape[0]),
            n_edges=edge_count,
        )"""
content = content.replace(old_from_adj, new_from_adj)

# 3. Update IncrementalSEState init to use node_cuts
old_init_cut = """        self.cut = self.volume - internal_twice
        self.cut[np.abs(self.cut) < 1e-10] = 0.0"""
new_init_cut = """        cluster_node_cuts = np.zeros(self.capacity, dtype=float)
        for node in range(graph.n_nodes):
            cid = int(self.labels[node])
            cluster_node_cuts[cid] += graph.node_cuts[node]
        self.cut = cluster_node_cuts - internal_twice
        self.cut[np.abs(self.cut) < 1e-10] = 0.0"""
content = content.replace(old_init_cut, new_init_cut)

# 4. Update IncrementalSEState move_delta to use node_cuts
old_move_delta = """        node_degree = self.graph.degrees[node]
        node_degree_log_degree = self.node_degree_log_degree[node]"""
new_move_delta = """        node_degree = self.graph.degrees[node]
        node_cut = self.graph.node_cuts[node]
        node_degree_log_degree = self.node_degree_log_degree[node]"""
content = content.replace(old_move_delta, new_move_delta)

old_new_cut_A = "new_cut_A = cut_A - node_degree + 2.0 * weight_to_A_without_node"
new_new_cut_A = "new_cut_A = cut_A - node_cut + 2.0 * weight_to_A_without_node"
content = content.replace(old_new_cut_A, new_new_cut_A)

old_new_cut_B = "new_cut_B = cut_B + node_degree - 2.0 * weight_to_B"
new_new_cut_B = "new_cut_B = cut_B + node_cut - 2.0 * weight_to_B"
content = content.replace(old_new_cut_B, new_new_cut_B)

# 5. Update apply_move to use node_cuts
old_apply_move = """        node_degree = self.graph.degrees[node]
        node_degree_log_degree = self.node_degree_log_degree[node]"""
new_apply_move = """        node_degree = self.graph.degrees[node]
        node_cut = self.graph.node_cuts[node]
        node_degree_log_degree = self.node_degree_log_degree[node]"""
content = content.replace(old_apply_move, new_apply_move)

old_apply_cut_current = "self.cut[current] += 2.0 * weight_to_current_without_node - node_degree"
new_apply_cut_current = "self.cut[current] += 2.0 * weight_to_current_without_node - node_cut"
content = content.replace(old_apply_cut_current, new_apply_cut_current)

old_apply_cut_target = "self.cut[target] += node_degree - 2.0 * weight_to_target"
new_apply_cut_target = "self.cut[target] += node_cut - 2.0 * weight_to_target"
content = content.replace(old_apply_cut_target, new_apply_cut_target)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
