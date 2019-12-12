from texttable import Texttable
import numpy as np
import networkx as nx
import pickle
import time
import random
from itertools import permutations, combinations
from sklearn.preprocessing import normalize
from scipy import sparse
import pdb
import os
import argparse


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


# def feature_calculator_pw(args):
#     """
#     Calculating the feature tensor.
#     :param args: Arguments object.
#     :param graph: NetworkX graph.
#     :return target_matrices: Target tensor.
#     """
#     try:
#         print('Loading exiting train data')
#         G = nx.load_gpickle(args.data_dir+args.graph_name+"/train_"+str(args.idx)+".gpickle")
#     except ExplicitException:
#         try:
#             print('Loading exiting train data')
#             if args.directed:
#                 test_neg_arr = np.load(open(graph_path+graph_name+'/test.directed.neg.txt.npy', 'rb'))
#                 G = nx.from_numpy_matrix(a, create_using=nx.DiGraph)
#             else:
#                 test_neg_arr = np.load(open(graph_path+graph_name+'/test.neg.txt.npy', 'rb'))
#                 G = nx.from_numpy_matrix(a, create_using=nx.Graph)
#         except ExplicitException::
#             print('Generating train/test data from the original Graph')
#             G = nx.load_gpickle(args.data_dir+args.graph_name+"/graph.gpickle")
#             sample_train_test_Graph(G, args.data_dir, idx=args.idx, test_ratio=args.test_ratio, is_directed =args.is_directed)
#             G = nx.load_gpickle(args.data_dir+args.graph_name+"/train_"+str(args.idx)+".gpickle")
#
#     A = nx.to_scipy_sparse_matrix(G)
#     transit_mat = normalize(A, norm='l1', axis=1).T
#     T_matrices = [transit_mat]
#     T_mat = transit_mat
#     if args.window_size > 1:
#         for power in tqdm(range(args.window_size-1), desc="Adjacency matrix powers"):
#             T_mat = T_mat.dot(transit_mat)
#             T_matrices.append(T_mat)
#
#     return T_matrices


def get_lcc(G, is_directed=True):
    if is_directed:
        G2 = max(nx.weakly_connected_component_subgraphs(G), key=len)
    else:
        G2 = max(nx.connected_component_subgraphs(G), key=len)
    tdl_nodes = G2.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    G2 = nx.relabel_nodes(G2, nodeListMap, copy=True)
    return G2, nodeListMap


def sample_train_test_Graph(G, data_dir, test_ratio=0.5, is_directed=True):
    """
    test_ratio <= 0.5
    keep self-loops(eg.ppi) in train_pos. test_pos, train_neg, test_neg have no self-loops.
    """
    start_time = time.time()
    print('Original Graph', nx.info(G))
    G, nodeListMap = get_lcc(G, is_directed)
    print('LCC Graph', nx.info(G))
    edge_list, test_edge_list = list(G.edges()), []
    random.shuffle(edge_list)
    e, n = len(edge_list), len(G.nodes())
    test_e = int(e*test_ratio)
    train_e = e - test_e

    G_train, count = G.copy(), 0
    if count < test_e:
        if is_directed:
            for edge in edge_list:
                G_train.remove_edge(edge[0], edge[1])
                if nx.is_weakly_connected(G_train) and len(G_train.nodes()) == n and edge[0] != edge[1]:
                    test_edge_list.append(edge)
                    count += 1
                else:
                    G_train.add_edge(edge[0], edge[1])
                if count == test_e:
                    break

        else:
            for edge in edge_list:
                G_train.remove_edge(edge[0], edge[1])
                if nx.is_connected(G_train) and len(G_train.nodes()) == n:
                    test_edge_list.append(list(edge))
                    count += 1
                else:
                    G_train.add_edge(edge[0], edge[1])
                if count == test_e:
                    break
    if count < test_e:
        print('Test ratio is too large. Please lower your test ratio!')

    print("Train Graph", nx.info(G_train))
    print("The number of test edges", len(test_edge_list))

    nx.write_gpickle(G_train, data_dir + "/train.gpickle")
    np.save(os.path.join(data_dir, "train.txt.npy"), np.array(G_train.edges()))
    np.save(os.path.join(data_dir, "test.txt.npy"), np.array(test_edge_list))
    with open(os.path.join(data_dir, 'nodelistmap.pickle'), 'wb') as f:
        pickle.dump(nodeListMap, f, protocol=pickle.HIGHEST_PROTOCOL)

    ## for small graphs
    # if is_directed:
    #     edge_neg_list = list(set(permutations(np.arange(n),2))-set(edge_list))
    # else:
    #     edge_neg_list = list(set(combinations(np.arange(n),2))-set(edge_list))
    # idx_neg = np.random.choice(len(edge_neg_list), e)
    # np.save(data_dir+"/train_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[idx_neg[:train_e],:])
    # np.save(data_dir+"/test_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[idx_neg[train_e:train_e+test_e],:])

    count_e, edge_neg_list, edge_list = 0, [], set(edge_list)
    if is_directed:
        while count_e < e:
            i, j = np.random.randint(n), np.random.randint(n)
            if i != j and (i, j) not in edge_list and (i, j) not in edge_neg_list:
                edge_neg_list.append((i, j))
                count_e += 1
    else:
        while count_e < e:
            i, j = np.random.randint(n), np.random.randint(n)
            if i != j and (i, j) not in edge_list and (j, i) not in edge_list \
                    and (i, j) not in edge_neg_list and (j, i) not in edge_neg_list:
                edge_neg_list.append((i, j))
                count_e += 1
    np.save(os.path.join(data_dir, "train.neg.txt.npy"), np.array(edge_neg_list)[:train_e, :])
    if is_directed:
        np.save(os.path.join(data_dir, "test.directed.neg.txt.npy"), np.array(edge_neg_list)[train_e:train_e + test_e, :])
    else:
        np.save(os.path.join(data_dir, "test.neg.txt.npy"), np.array(edge_neg_list)[train_e:train_e + test_e, :])
    print("Used Time --- %s seconds ---" % (time.time() - start_time))

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Data')
    parser.add_argument('--data_dir', type=str, default='datasets/citeseer', help='Data dir')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='Data dir')
    args = parser.parse_args()
    print(args)

    filename = os.path.join(args.data_dir, 'graph.gpickle')
    G = nx.read_gpickle(filename)
    sample_train_test_Graph(G, args.data_dir, test_ratio=args.test_ratio, is_directed=G.is_directed())
