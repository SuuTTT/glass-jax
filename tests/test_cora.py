import time
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, to_undirected
import numpy as np

dataset = Planetoid(root='/tmp/dataset', name='Cora')
data = dataset[0]
edge_index = to_undirected(data.edge_index)
adj = to_dense_adj(edge_index)[0].numpy()
labels = data.y.numpy()
k = dataset.num_classes
print(f"Cora nodes: {adj.shape[0]}, classes: {k}")
