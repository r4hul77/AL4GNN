import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import argparse
import networkx as nx
from src.models.model import SAGE
from src.data import utils as ut
from plot_results import *

import os
from src.data import config as cnf
import warnings
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy

th.manual_seed(69)
np.random.seed(69)

## split the graph
def inductive_split(g):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = copy.deepcopy(g)
    return train_g, val_g, test_g

# train evaluate
def al_evaluate(model, test_nfeat, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0
    running_f1 = 0.0

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            temp_pred = th.argmax(batch_pred, dim=1)
            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            current_f1 = f1_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy(), average='weighted')
            running_f1 = running_f1 + ((1 / (step + 1)) * (current_f1 - running_f1))

            # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            # print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, running_f1

# test evaluate
def al_evaluate_test(model, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0
    running_f1 = 0.0

    # get intermediate output as node embeddings
    # activation = {}
    #
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #
    #     return hook

    # model.layers[2].fc_neigh.register_forward_hook(get_activation('layers[2].fc_neigh'))

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            # node_emb = activation['layers[2].fc_neigh']

            temp_pred = th.argmax(batch_pred, dim=1)
            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            current_f1 = f1_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy(), average='weighted')
            running_f1 = running_f1 + ((1 / (step + 1)) * (current_f1 - running_f1))

            # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            # print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, running_f1

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    th.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model):
    checkpoint = th.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, valid_loss_min.item()


# ============= get initial training set of graph ================
def get_basetraining_set(graph, n_classes, val_size, base_size):

    array_lst_class = []
    for cclass in range(n_classes):
        array_lst_class.append([])

    for node in graph.nodes():
        temp = graph.ndata['label'][node]
        array_lst_class[temp].append(node.item())

    rng = random.Random(69)
    array_lst_base_train = [rng.sample(array_lst_class[ind], base_size) for ind in range(n_classes)]

    array_lst_avail_val = []
    for ind in range(n_classes):
        array_lst_avail_val.append([elem for elem in array_lst_class[ind] if elem not in array_lst_base_train[ind]])

    array_lst_val = [rng.sample(array_lst_avail_val[ind], val_size) for ind in range(n_classes)]
    lst_val = [item for sublist in array_lst_val for item in sublist]

    array_lst_avail_test = []
    for ind in range(n_classes):
        array_lst_avail_test.append([elem for elem in array_lst_class[ind] if
                                     elem not in array_lst_base_train[ind] and elem not in array_lst_val[ind]])

    # array_lst_test = [rng.sample(array_lst_avail_test[ind], test_size) for ind in range(n_classes)]
    # lst_test = [item for sublist in array_lst_class for item in sublist]

    # array_lst_avail_train = []
    # for ind in range(n_classes):
    #     array_lst_avail_train.append([elem for elem in array_lst_class[ind] if elem not in array_lst_base_train[ind]
    #                                   and elem not in array_lst_val[ind] and elem not in array_lst_test[ind]])

    array_lst_base_train = [item for sublist in array_lst_base_train for item in sublist]
    array_lst_avail_train = [item for sublist in array_lst_avail_test for item in sublist]

    return lst_val, array_lst_base_train, array_lst_avail_train

# ============= get validation test mask in graph ================
def get_valtest_mask(graph, lst_val, maskname):

        n_nodes = graph.num_nodes()
        val_mask = th.zeros(n_nodes, dtype=th.bool)
        # test_mask = th.zeros(n_nodes, dtype=th.bool)

        val_mask[lst_val] = True
        # test_mask[lst_test] = True

        graph.ndata[maskname] = val_mask
        # graph.ndata['test_mask'] = test_mask

# ============= centrality sampling ================
def get_centralitysampling(args, device, gsample, best_model_path, lst_unlabeled_nodes):

    gsample_new = copy.deepcopy(gsample)

    # update unlabeled mask of sample graph
    n_nodes = gsample_new.num_nodes()
    test_mask = th.zeros(n_nodes, dtype=th.bool)
    test_mask[lst_unlabeled_nodes] = True
    gsample_new.ndata['test_mask'] = test_mask

    test_g = copy.deepcopy(gsample_new)
    test_nfeat = test_g.ndata.pop('features')
    test_labels = test_g.ndata.pop('labels')

    test_nfeat = test_nfeat.to(device)
    test_labels = test_labels.to(device)

    uc_data = test_g, test_nfeat, test_labels

    y_predprob = run_test_uncertsample(args, device, uc_data, best_model_path)

    # ============= entropy method works better than class wise entropy =============
    y_logprob = np.log(y_predprob)
    y_pred = np.multiply(y_predprob, y_logprob)
    entropy = np.sum(y_pred, axis=1)
    entropy = entropy*(-1)
    node_sampled = lst_unlabeled_nodes[np.argmax(entropy)]

    #============ class entropy method =======

    # temp_ypredprob = np.max(y_predprob, axis=1)
    # node_sampled = lst_unlabeled_nodes[np.argmin(temp_ypredprob)]

    return [node_sampled]

# ============= uncertainty sampling ================
def get_uncertsampling(args, device, gsample, best_model_path, lst_unlabeled_nodes):

    gsample_new = copy.deepcopy(gsample)

    # update unlabeled mask of sample graph
    n_nodes = gsample_new.num_nodes()
    test_mask = th.zeros(n_nodes, dtype=th.bool)
    test_mask[lst_unlabeled_nodes] = True
    gsample_new.ndata['test_mask'] = test_mask

    test_g = copy.deepcopy(gsample_new)
    test_nfeat = test_g.ndata.pop('features')
    test_labels = test_g.ndata.pop('labels')

    test_nfeat = test_nfeat.to(device)
    test_labels = test_labels.to(device)

    uc_data = test_g, test_nfeat, test_labels

    y_predprob = run_test_uncertsample(args, device, uc_data, best_model_path)

    # ============= entropy method works better than class wise entropy =============
    y_logprob = np.log(y_predprob)
    y_pred = np.multiply(y_predprob, y_logprob)
    entropy = np.sum(y_pred, axis=1)
    entropy = entropy*(-1)
    node_sampled = lst_unlabeled_nodes[np.argmax(entropy)]

    #============ class entropy method =======

    # temp_ypredprob = np.max(y_predprob, axis=1)
    # node_sampled = lst_unlabeled_nodes[np.argmin(temp_ypredprob)]

    return [node_sampled]

# ============= sampled new nodes from unlabeled pool using query strategy ================
def get_queriedsample(args, device, gsample, best_model_path, lst_unlabeled_nodes, query_batch_size, query_strategy, epsilon=0.1):

    randno = random.uniform(0, 1)

    if randno >= epsilon:
        if query_strategy == "random_sampling":
            node_sampled = random.sample(lst_unlabeled_nodes, query_batch_size)

        elif query_strategy == "uncertainty_sampling":
            node_sampled = get_uncertsampling(args, device, gsample, best_model_path, lst_unlabeled_nodes)

        elif query_strategy == "betweenness_sampling":
            node_sampled = get_centralitysampling(args, device, gsample, best_model_path, lst_unlabeled_nodes)

    else:
        node_sampled = random.sample(lst_unlabeled_nodes, query_batch_size)
        # array_lst_sampled = [random.sample(lst_unlabeled_nodes[ind], query_batch_size) for ind in range(n_classes)]

    return node_sampled

# ============= get new updated train test mask in graph ================
def update_train_mask(graph, lst_labeled_nodes, lst_unlabeled_nodes, node_sampled):

    lst_labeled_nodes.extend(node_sampled)

    n_nodes = graph.num_nodes()
    train_mask = th.zeros(n_nodes, dtype=th.bool)

    train_mask[lst_labeled_nodes] = True

    graph.ndata['train_mask'] = train_mask

    try:
        lst_unlabeled_nodes.remove(node_sampled.pop())
    except:
        pass

    return lst_labeled_nodes, lst_unlabeled_nodes

#### Entry point

def run(args, device, data, checkpoint_path, best_model_path):

    # Unpack data
    # train_losslist = []
    # val_losslist = []

    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels, g = data

    in_feats = train_nfeat.shape[1]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]

    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    # test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden_dim, n_classes, args.num_layers, F.relu, args.dropout)
    print("== # model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)

    # == customize loss function
    # custom loss function
    # class_weights = th.FloatTensor(weights).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = nn.BCELoss(weight = class_weights)
    # loss_fcn = weighted_binary_cross_entropy()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(train_g, train_nid, sampler)

    # validata dataloader
    valdataloader = get_dataloader(val_g, val_nid, sampler)

    # Training loop
    valid_loss_min = np.Inf

    for epoch in range(args.num_epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        # tic_step = time.time()
        # epoch_loss = []
        model.train()

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            loss = loss_fcn(batch_pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        train_acc, train_loss, train_f1 = al_evaluate(model, train_nfeat, train_labels, device, dataloader, loss_fcn)
        val_acc, val_loss, val_f1 = al_evaluate(model, val_nfeat, val_labels, device, valdataloader, loss_fcn)

        print('Epoch: {} \tTraining acc: {:.6f} \tValidation acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_acc,
            val_acc,
            train_loss,
            val_loss
            ))

        checkpoint = {
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
        }

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # TODO: save the model if validation loss has decreased

        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, val_loss))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = val_loss

    return train_acc, train_f1, val_acc, val_f1

def run_test(args, device, gsample, best_model_path):

    # Unpack data
    # train_losslist = []
    # val_losslist = []
    # n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    # val_nfeat, val_labels, test_nfeat, test_labels, g = data

    gsample_new = copy.deepcopy(gsample)

    # update unlabeled mask of sample graph
    n_nodes = gsample_new.num_nodes()
    test_mask = th.ones(n_nodes, dtype=th.bool)
    # test_mask[lst_unlabeled_nodes] = True
    gsample_new.ndata['test_mask'] = test_mask

    test_g = copy.deepcopy(gsample_new)
    test_nfeat = test_g.ndata.pop('features')
    test_labels = test_g.ndata.pop('labels')

    test_nfeat = test_nfeat.to(device)
    test_labels = test_labels.to(device)

    # uc_data = test_g, test_nfeat, test_labels
    #---- test on test mask nodes--
    in_feats = test_nfeat.shape[1]
    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]
    # test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        test_nid = test_nid.to(device)
        # copy only the csc to the GPU
        test_g = test_g.formats(['csc'])
        test_g = test_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=test_g.num_nodes(),
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden_dim, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)

    # == customize loss function
    # custom loss function
    # class_weights = th.FloatTensor(weights).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = nn.BCELoss(weight = class_weights)
    # loss_fcn = weighted_binary_cross_entropy()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, valid_loss_min = load_ckp(best_model_path, model)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(test_g, test_nid, sampler)

    model.eval()

    test_acc, test_loss, test_f1 = al_evaluate_test(model, test_labels, device, dataloader, loss_fcn)

    print('Test acc: {:.6f} \tTess Loss: {:.6f}'.format(test_acc, test_loss))

    return test_acc, test_loss, test_f1

    # plt.plot(np.linspace(1, args.num_epochs, args.num_epochs).astype(int), train_losslist)
    # plt.figure(2)
    # plt.plot(np.linspace(1, args.num_epochs, args.num_epochs).astype(int), val_losslist)

def run_test_uncertsample(args, device, data, best_model_path):

    # Unpack data
    # train_losslist = []
    # val_losslist = []

    test_g, test_nfeat, test_labels = data

    in_feats = test_nfeat.shape[1]

    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        test_nid = test_nid.to(device)
        # copy only the csc to the GPU
        test_g = test_g.formats(['csc'])
        test_g = test_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=test_g.num_nodes(),
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden_dim, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)

    # == customize loss function

    model, valid_loss_min = load_ckp(best_model_path, model)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(test_g, test_nid, sampler)

    model.eval()

    # probability prediction

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

    model.train()

    y_pred= batch_pred.cpu().detach().numpy()

    y_predprob = []
    for count in range(y_pred.shape[0]):
        y_predprob.append(softmax(y_pred[count,:]))

    return np.array(y_predprob)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--n_class', type=int, default = 1)
    argparser.add_argument('--n_queries', type=int, default = 3)
    argparser.add_argument('--query_batch_size', type=int, default = 10)
    argparser.add_argument('--val_size', type=int, default = 20)
    # argparser.add_argument('--test_size', type=int, default = 100)
    argparser.add_argument('--base_size', type=int, default = 50)
    # argparser.add_argument('--filepath', default = os.path.join(cnf.datapath, "pubmed_weighted.gpickle"))
    argparser.add_argument('--checkpointpath', default = os.path.join(cnf.modelpath, "current_checkpoint_pubmed.pt"))
    argparser.add_argument('--bestmodelpath', default = os.path.join(cnf.modelpath, "pubmed_random.pt"))
    argparser.add_argument('--resultscsvpath', default = os.path.join(cnf.resultspath, "ALResultsdf_pubmed_b50_q100_uncertsamp.csv"))
    argparser.add_argument('--jpegpath', default=os.path.join(cnf.modelpath, "Graph"))
    argparser.add_argument('--query_strategy', default = "uncertainty_sampling")
    # argparser.add_argument('--test_batch_size', type=int, default= 19700)

    argparser.add_argument('--num-epochs', type=int, default = 15)
    argparser.add_argument('--hidden_dim', type=int, default = 48)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='8,10')
    argparser.add_argument('--batch-size', type=int, default= 10)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.2)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    # argparser.add_argument('--inductive', action='store_true',
    #                        help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")

    argparser.add_argument('--data_set', type=int, default=2)


    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # ============= load graph ================

    # ============= read graph from dgl library ========

    query_strategies = ['random_sampling', 'uncertainty_sampling']

    for query_strategy in query_strategies:

        for data_set in range(3):

            if(data_set == 0):
                dataset = dgl.data.PubmedGraphDataset()
                dataset_name = "PubMed"

            elif(data_set == 1):
                dataset = dgl.data.CoraGraphDataset()
                dataset_name = "Cora"
            elif(data_set==2):
                dataset = dgl.data.AmazonCoBuyComputerDataset()
                dataset_name = "Amazon"
            else:
                print("Unknown argument sent setting default dataset to Pub Med")
                dataset = dgl.data.PubmedGraphDataset()
                dataset_name = "PubMed"

            g = dataset[0]

            results_folder = dataset_name + "_" + query_strategy + "q{}".format(args.n_queries)

            results_folder_path = os.path.join(cnf.resultspath, results_folder)

            start_time = time.time()

            if(not os.path.isdir(results_folder_path)):
                os.mkdir(results_folder_path)

            #==== read from weighted graph in netwrokx

            # graph = nx.read_gpickle(args.filepath)
            # graph = nx.to_directed(graph)
            #
            # g = dgl.from_networkx(graph, node_attrs=['feature', 'label'])
            #
            # g.ndata['feat'] = g.ndata['feature']
            # g.ndata['label'] = g.ndata['label'].long()

            g.ndata['features'] = g.ndata['feat']
            g.ndata['labels'] = g.ndata['label']

            gsample = copy.deepcopy(g)

            # ============= get validation test mask in graph ================

            n_classes = dataset.num_classes

            lst_val, lst_labeled_nodes, lst_unlabeled_nodes = get_basetraining_set(g, n_classes, args.val_size, args.base_size)

            # mask val nodes
            get_valtest_mask(g, lst_val, 'val_mask')

            # loop for query
            train_size_list = []
            train_acc_list = []
            train_f1_list = []
            test_acc_list = []
            test_f1_list = []
            node_sampled_list = []

            for cquery in range(args.n_queries+1):

                if cquery == 0:
                    node_sampled = []
                else:
                    node_sampled = get_queriedsample(args, device, gsample, args.bestmodelpath, lst_unlabeled_nodes, args.query_batch_size, query_strategy, epsilon=0.0)

                node_sampled_list.append(node_sampled)

                # update train mask, labled training data and unlabeled training data
                lst_labeled_nodes, lst_unlabeled_nodes = update_train_mask(g, lst_labeled_nodes,
                                                                                lst_unlabeled_nodes, node_sampled)

                train_g, val_g, test_g = inductive_split(g)

                train_nfeat = train_g.ndata.pop('features')
                val_nfeat = val_g.ndata.pop('features')
                test_nfeat = test_g.ndata.pop('features')
                train_labels = train_g.ndata.pop('labels')
                val_labels = val_g.ndata.pop('labels')
                test_labels = test_g.ndata.pop('labels')

                print("train, val nodes", train_nfeat.shape, val_nfeat.shape)
                print("input graph size :", train_nfeat.shape)
                print("query no", cquery)

                if not args.data_cpu:
                    train_nfeat = train_nfeat.to(device)
                    val_nfeat = val_nfeat.to(device)
                    test_nfeat = test_nfeat.to(device)
                    train_labels = train_labels.to(device)
                    val_labels = val_labels.to(device)
                    test_labels = test_labels.to(device)

                # Pack data
                data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
                       val_nfeat, val_labels, test_nfeat, test_labels, g

                # ============= train model on current labled lst_labeled_nodes data ================

                train_acc, train_f1, val_acc, val_f1 = run(args, device, data, args.checkpointpath, args.bestmodelpath)

                # ============= evaluate model on test data ================

                test_acc, test_loss, test_f1 = run_test(args, device, gsample, args.bestmodelpath)

                train_size_list.append(train_nfeat.shape[0])
                train_acc_list.append(train_acc)
                train_f1_list.append(train_f1)

                test_acc_list.append(test_acc)
                test_f1_list.append(test_f1)

            # ============= save final results in dataframe ================
            Resultsdf = pd.DataFrame()

            Resultsdf['train_size'] = train_size_list
            Resultsdf['train_acc'] = train_acc_list
            Resultsdf['train_f1'] = train_f1_list
            Resultsdf['test_acc'] = test_acc_list
            Resultsdf['test_f1'] = test_f1_list

            Resultsdf.to_csv(os.path.join(results_folder_path, "results.csv"))

            plot_results(Resultsdf, query_strategy, dataset_name, results_folder_path)

            end_time = time.time()
            total_time = end_time - start_time

            with(open(os.path.join(results_folder_path, "performance.time"), "w")) as f:
                f.write(str(total_time))










