from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import coo_matrix

dataset = NodePropPredDataset(name = 'ogbn-arxiv')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: library-agnostic graph object

trainset = train_idx.astype(np.int32)
trainset.tofile('./paper100M/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./paper100M/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./paper100M/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./paper100M/'+'labels')
features = graph['node_feat'].astype(np.float32)
features.tofile('./paper100M/'+'features')


# Edge index in COO format
edge_index = graph['edge_index']
num_nodes = graph['num_nodes']
print(num_nodes)
# Convert to COO matrix
coo = coo_matrix((edge_index[1], (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

# Convert to CSR format
csr = coo.tocsr()

# Get the CSR row and col arrays
edge_src = (csr.indptr).astype(np.int64)
edge_src.tofile('./paper100M/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./paper100M/'+'edge_dst')

xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/paper100m_xtraformat')

