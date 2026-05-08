import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

# Add connectedness refinement
refinement_code = """        k = int(labels.max()) + 1
        if k == current_adj.shape[0] or k == 1:
            projections.append(labels)
            break
            
        import scipy.sparse.csgraph as csgraph
        new_labels = np.zeros_like(labels)
        next_label = 0
        for i in range(k):
            mask = (labels == i)
            if not np.any(mask): continue
            sub_adj = current_adj[mask][:, mask]
            n_comp, comp_labels = csgraph.connected_components(sub_adj, directed=False)
            new_labels[mask] = comp_labels + next_label
            next_label += n_comp
            
        labels = canonicalize_labels(new_labels)
        k = int(labels.max()) + 1
        
        projections.append(labels)"""

old_code = """        k = int(labels.max()) + 1
        if k == current_adj.shape[0] or k == 1:
            projections.append(labels)
            break
            
        projections.append(labels)"""

content = content.replace(old_code, refinement_code)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
