import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

old_sig = """def multi_level_local_move(
    adj: np.ndarray,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, float]:"""

new_sig = """def multi_level_local_move(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    return_hierarchy: bool = False,
):"""

content = content.replace(old_sig, new_sig)

old_return = """    final_labels = canonicalize_labels(final_labels)
    entropy = structural_entropy(adj, final_labels)
    return final_labels, entropy"""

new_return = """    final_labels = canonicalize_labels(final_labels)
    entropy = structural_entropy(adj, final_labels)
    if return_hierarchy:
        return final_labels, entropy, projections
    return final_labels, entropy"""

content = content.replace(old_return, new_return)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)
