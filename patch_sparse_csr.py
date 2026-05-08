import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

# Replace from_adjacency with from_csr
new_from_csr = """    @classmethod
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

old_from_adj = """    @classmethod
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

content = content.replace(old_from_adj, new_from_csr)

old_local_move = """def local_move_incremental(
    adj: np.ndarray,
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
) -> tuple[np.ndarray, float]:
    \"\"\"Run sparse incremental node-move SE local search.\"\"\"

    graph = SparseGraph.from_adjacency(adj)"""

content = content.replace(old_local_move, new_local_move)


old_multi_level = """def multi_level_local_move(
    adj: np.ndarray,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
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
        )
        
        k = int(labels.max()) + 1
        if k == current_adj.shape[0] or k == 1:
            projections.append(labels)
            break
            
        projections.append(labels)
        
        row = np.arange(len(labels))
        col = labels
        data = np.ones(len(labels), dtype=current_adj.dtype if hasattr(current_adj, 'dtype') else np.float64)
        S = sp.csr_matrix((data, (row, col)), shape=(len(labels), k))
        A_sparse = sp.csr_matrix(current_adj)
        current_adj = S.T @ A_sparse @ S
        
        current_labels = np.arange(k, dtype=np.int32)
        
    final_labels = projections[-1]"""

content = content.replace("current_adj = (S.T @ A_sparse @ S).toarray()", "current_adj = S.T @ A_sparse @ S")

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
