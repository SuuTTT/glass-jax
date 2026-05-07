import sys
import os
import time
import numpy as np

# Add official SEP to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../official_baselines/SEP')))
from SEPN.codingTree import PartitionTree

def run_official_sep(adj, k, seed=42):
    start = time.time()
    # Official SEP code doesn't use seed for greedy search, it's deterministic
    tree = PartitionTree(adj_matrix=adj)
    # k=2 in original, but we pass our k
    # Their build_coding_tree is complex. We'll use mode='v1' for direct k-tree
    try:
        tree.build_coding_tree(k=k, mode='v1')
        labels = np.zeros(adj.shape[0], dtype=int)
        cluster_idx = 0
        for _, node in tree.tree_node.items():
            if not node.merged:
                for n in node.partition:
                    labels[n] = cluster_idx
                cluster_idx += 1
    except Exception as e:
        print(f"Official SEP failed: {e}")
        labels = np.zeros(adj.shape[0], dtype=int)
    duration = time.time() - start
    return labels, duration

if __name__ == "__main__":
    from tests.benchmark import get_sbm
    adj, gt, k = get_sbm(150, 3, 0.3, 0.05, 42)
    labels, dur = run_official_sep(adj, k)
    print(f"Official SEP Duration: {dur:.4f}")
    from sklearn.metrics import adjusted_rand_score
    print(f"Official SEP ARI: {adjusted_rand_score(gt, labels)}")
