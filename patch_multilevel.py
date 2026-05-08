import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

multi_level_code = """
import scipy.sparse as sp

def multi_level_local_move(
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
        data = np.ones(len(labels), dtype=current_adj.dtype)
        S = sp.csr_matrix((data, (row, col)), shape=(len(labels), k))
        A_sparse = sp.csr_matrix(current_adj)
        current_adj = (S.T @ A_sparse @ S).toarray()
        
        current_labels = np.arange(k, dtype=np.int32)
        
    final_labels = projections[-1]
    for i in range(len(projections) - 2, -1, -1):
        final_labels = final_labels[projections[i]]
        
    final_labels = canonicalize_labels(final_labels)
    entropy = structural_entropy(adj, final_labels)
    return final_labels, entropy
"""

# Insert multi_level_local_move before multistart_incremental_se_heuristic
old_multistart_def = "def multistart_incremental_se_heuristic("
content = content.replace(old_multistart_def, multi_level_code + "\n\n" + old_multistart_def)

# Change local_move_incremental to multi_level_local_move inside multistart
old_call = """        candidate_labels, entropy = local_move_incremental(
            adj,
            init_labels=labels,
            max_passes=max_passes,
            seed=seed + i,
            allow_new_cluster=True,
        )"""
new_call = """        candidate_labels, entropy = multi_level_local_move(
            adj,
            init_labels=labels,
            max_passes=max_passes,
            seed=seed + i,
        )"""
content = content.replace(old_call, new_call)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
