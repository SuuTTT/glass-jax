import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

# 1. Update SparseGraph to include node_degree_log_degree
old_sg = """class SparseGraph:
    \"\"\"Neighbor-list view of a non-negative undirected graph.\"\"\"

    neighbors: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    degrees: np.ndarray
    node_cuts: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int"""

new_sg = """class SparseGraph:
    \"\"\"Neighbor-list view of a non-negative undirected graph.\"\"\"

    neighbors: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    degrees: np.ndarray
    node_cuts: np.ndarray
    node_degree_log_degree: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int"""

content = content.replace(old_sg, new_sg)

# 2. Update from_adjacency to compute or accept it
old_from = """    @classmethod
    def from_adjacency(cls, adj) -> "SparseGraph":
        import scipy.sparse as sp
        if sp.issparse(adj):
            matrix = adj.tocsr()
            matrix = 0.5 * (matrix + matrix.T)
        else:
            matrix = np.asarray(adj, dtype=float)
            matrix = 0.5 * (matrix + matrix.T)
            matrix = sp.csr_matrix(matrix)
            
        degrees = np.asarray(matrix.sum(axis=1)).flatten()
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        node_cuts = np.asarray(matrix.sum(axis=1)).flatten()
        
        neighbors = []
        weights = []
        edge_count = 0
        for node in range(matrix.shape[0]):
            start = matrix.indptr[node]
            end = matrix.indptr[node+1]
            idx = matrix.indices[start:end]
            vals = matrix.data[start:end]
            
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals.astype(float, copy=False))
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

new_from = """    @classmethod
    def from_adjacency(cls, adj, node_degree_log_degree=None) -> "SparseGraph":
        import scipy.sparse as sp
        if sp.issparse(adj):
            matrix = adj.tocsr()
            matrix = 0.5 * (matrix + matrix.T)
        else:
            matrix = np.asarray(adj, dtype=float)
            matrix = 0.5 * (matrix + matrix.T)
            matrix = sp.csr_matrix(matrix)
            
        degrees = np.asarray(matrix.sum(axis=1)).flatten()
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        node_cuts = np.asarray(matrix.sum(axis=1)).flatten()
        
        if node_degree_log_degree is None:
            node_degree_log_degree = np.zeros_like(degrees)
            positive = degrees > 1e-12
            node_degree_log_degree[positive] = degrees[positive] * np.log2(degrees[positive])
        
        neighbors = []
        weights = []
        edge_count = 0
        for node in range(matrix.shape[0]):
            start = matrix.indptr[node]
            end = matrix.indptr[node+1]
            idx = matrix.indices[start:end]
            vals = matrix.data[start:end]
            
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals.astype(float, copy=False))
            edge_count += int(np.sum(idx > node))
            
        return cls(
            neighbors=tuple(neighbors),
            weights=tuple(weights),
            degrees=degrees.astype(float, copy=False),
            node_cuts=node_cuts.astype(float, copy=False),
            node_degree_log_degree=np.asarray(node_degree_log_degree, dtype=float),
            volume=float(degrees.sum()),
            n_nodes=int(matrix.shape[0]),
            n_edges=edge_count,
        )"""

content = content.replace(old_from, new_from)

# 3. Update IncrementalSEState to use graph.node_degree_log_degree
old_init_log = """        self.node_degree_log_degree = np.zeros(graph.n_nodes, dtype=float)

        positive = graph.degrees > eps
        self.node_degree_log_degree[positive] = graph.degrees[positive] * np.log2(graph.degrees[positive])

        for node, cluster in enumerate(self.labels):"""

new_init_log = """        self.node_degree_log_degree = graph.node_degree_log_degree.copy()

        for node, cluster in enumerate(self.labels):"""

content = content.replace(old_init_log, new_init_log)

# 4. Pass node_degree_log_degree along in multi_level_local_move
old_ml = """def multi_level_local_move(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    return_hierarchy: bool = False,
):
    rng = np.random.default_rng(seed)
    
    current_adj = adj
    current_labels = init_labels
    
    projections = []
    
    while True:
        labels, _ = local_move_incremental(
            current_adj,
            init_labels=current_labels,
            max_passes=max_passes,
            seed=int(rng.integers(1<<30)),
            allow_new_cluster=True,
        )"""

new_ml = """def multi_level_local_move(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    return_hierarchy: bool = False,
):
    rng = np.random.default_rng(seed)
    
    current_adj = adj
    current_labels = init_labels
    current_dlogd = None
    
    projections = []
    
    while True:
        labels, _ = local_move_incremental(
            current_adj,
            node_degree_log_degree=current_dlogd,
            init_labels=current_labels,
            max_passes=max_passes,
            seed=int(rng.integers(1<<30)),
            allow_new_cluster=True,
        )"""

content = content.replace(old_ml, new_ml)

old_proj = """        row = np.arange(len(labels))
        col = labels
        data = np.ones(len(labels), dtype=current_adj.dtype if hasattr(current_adj, 'dtype') else np.float64)
        S = sp.csr_matrix((data, (row, col)), shape=(len(labels), k))
        A_sparse = sp.csr_matrix(current_adj)
        current_adj = S.T @ A_sparse @ S
        
        current_labels = np.arange(k, dtype=np.int32)"""

new_proj = """        row = np.arange(len(labels))
        col = labels
        data = np.ones(len(labels), dtype=current_adj.dtype if hasattr(current_adj, 'dtype') else np.float64)
        S = sp.csr_matrix((data, (row, col)), shape=(len(labels), k))
        A_sparse = sp.csr_matrix(current_adj)
        
        # Calculate true degrees of the current graph before projection
        # This is needed because current_adj has 0 on diagonal
        temp_graph = SparseGraph.from_adjacency(current_adj, current_dlogd)
        if current_dlogd is None:
            current_dlogd = temp_graph.node_degree_log_degree
            
        current_adj = S.T @ A_sparse @ S
        
        # Project node_degree_log_degree by summing it for the new super-nodes
        new_dlogd = np.zeros(k, dtype=float)
        np.add.at(new_dlogd, labels, current_dlogd)
        current_dlogd = new_dlogd
        
        current_labels = np.arange(k, dtype=np.int32)"""

content = content.replace(old_proj, new_proj)

old_local_move = """def local_move_incremental(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    allow_new_cluster: bool = True,
) -> tuple[np.ndarray, float]:
    \"\"\"Run sparse incremental node-move SE local search.\"\"\"

    graph = SparseGraph.from_adjacency(adj)"""

new_local_move = """def local_move_incremental(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    allow_new_cluster: bool = True,
    node_degree_log_degree=None,
) -> tuple[np.ndarray, float]:
    \"\"\"Run sparse incremental node-move SE local search.\"\"\"

    graph = SparseGraph.from_adjacency(adj, node_degree_log_degree)"""

content = content.replace(old_local_move, new_local_move)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
