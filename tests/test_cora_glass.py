import time
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, to_undirected
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tests.benchmark import run_glass_jax_multistart
from glass.objectives.structural_entropy import two_dimensional_structural_entropy

dataset = Planetoid(root='/tmp/dataset', name='Cora')
data = dataset[0]
edge_index = to_undirected(data.edge_index)
adj = to_dense_adj(edge_index)[0].numpy()
labels = data.y.numpy()
k = dataset.num_classes

start = time.time()
pred, dur = run_glass_jax_multistart(adj, k, two_dimensional_structural_entropy, n_iters=100, n_starts=2)
print(f"Glass-SE on Cora: {dur:.2f}s")
from sklearn.metrics import adjusted_rand_score
print(f"ARI: {adjusted_rand_score(labels, pred):.3f}")
