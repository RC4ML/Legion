from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import coo_matrix

# root = '/home/wzq/datasets/OGB' can be replaced to the path where you store the dataset
dataset = NodePropPredDataset(name = 'ogbn-products',root='/home/wzq/datasets/OGB')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: library-agnostic graph object

trainset = train_idx.astype(np.int32)
trainset.tofile('./products/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./products/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./products/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./products/'+'labels')
features = graph['node_feat'].astype(np.float32)
features.tofile('./products/'+'features')


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
edge_src.tofile('./products/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./products/'+'edge_dst')

xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/products_xtraformat')

