import dgl
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
from src.data import config as cnf
import numpy as np

class PLCgraphDataset(DGLDataset):

    def __init__(self, train_size, val_size):
        super().__init__(name='PLC graph')
        self.train_size = train_size
        self.val_size = val_size
        # print("train size", self.train_size)

    def process(self):

        filepath = cnf.datapath + "\\pubmed_weighted.gpickle"

        g = nx.read_gpickle(filepath)
        g = nx.to_directed(g)

        self.graph = dgl.from_networkx(g, node_attrs=['feature','label'], edge_attrs=['weight'])

        self.graph.ndata['feat'] = self.graph.ndata['feature']

        self.graph.ndata['label'] = self.graph.ndata['label']

        # generate mask for testing nodes
        lst_c1=[]
        lst_c2=[]
        lst_c3=[]

        for countid,(node,d) in enumerate(g.nodes(data=True)):

            if d['label'] == 0:
                lst_c1.append(countid)

            elif d['label'] == 1:
                lst_c2.append(countid)

            elif d['label'] == 2:
                lst_c3.append(countid)

        lst_c = list(np.random.choice(lst_c1, 20)) + list(np.random.choice(lst_c2,20)) + list(np.random.choice(lst_c3,20))

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.graph.num_nodes()
        n_train = int(n_nodes * 0.01)
        n_val = int(n_nodes * 0.01)

        # TODO: ADD NEW MASK FOR TRAINING SET START WITH 20%-50% IN STEPS OF 5%
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_almask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        train_almask[lst_c] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['train_Almask'] = train_almask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 3

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def load_plcgraph(filepath, train_ratio=0.005, valid_ratio=0.005):
    # load PLC data
    data = PLCgraphDataset(train_size=0.8, val_size=0.1)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    return g, data.num_classes

# CHNAGES
def inductive_split(g):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    train_g = g.subgraph(g.ndata['train_mask'])
    train_alg = g.subgraph(g.ndata['train_Almask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g, train_alg
