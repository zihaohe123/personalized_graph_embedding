from texttable import Texttable
import numpy as np
import networkx as nx
import pickle
import time
import random
from itertools import permutations, combinations



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




def get_lcc(G, is_directed = True):
    if is_directed: 
        G2 = max(nx.weakly_connected_component_subgraphs(G), key=len)
    else: 
        G2 = max(nx.connected_component_subgraphs(G), key=len)
    tdl_nodes = G2.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    G2 = nx.relabel_nodes(G2, nodeListMap, copy=True)
    return G2, nodeListMap




def sample_train_test_Graph(G, data_dir, idx=0, test_ratio=0.5, is_directed =True):
    """
    test_ratio <= 0.5
    keep self-loops(eg.ppi) in train_pos. test_pos, train_neg, test_neg have no self-loops. 
    """
    start_time = time.time()
    print('Original Graph', nx.info(G))
    G = get_lcc(G, is_directed)[0]
    print('LCC Graph', nx.info(G))
    edge_list, test_edge_list = list(G.edges()), []
    random.shuffle(edge_list)
    e, n = len(edge_list), len(G.nodes())
    test_e = int(e*(test_ratio))
    train_e = e - test_e

    G_train, count = G.copy(), 0
    if is_directed:
        for edge in edge_list:
            G_train.remove_edge(edge[0],edge[1])
            if nx.is_weakly_connected(G_train) and len(G_train.nodes())==n and edge[0]!= edge[1]:
                test_edge_list.append(edge)
                count+=1
            else: G_train.add_edge(edge[0],edge[1])
            if count == test_e: break
        
    else:
        for edge in edge_list:
            G_train.remove_edge(edge[0],edge[1])
            if nx.is_connected(G_train) and len(G_train.nodes())==n:
                test_edge_list.append(list(edge))
                count+=1
            else: G_train.add_edge(edge[0],edge[1])
            if count == test_e: break
                
    if count < test_e:
        print('Test ratio is too large. Please lower your test ratio!')
        
    print("Train Graph", nx.info(G_train))
    print("The number of test edges", len(test_edge_list))
                
    nx.write_gpickle(G_train, data_dir+"/train_"+str(idx)+".gpickle")
    np.save(data_dir+"/train_"+str(idx)+".txt.npy",np.array(G_train.edges()))
    np.save(data_dir+"/test_"+str(idx)+".txt.npy", np.array(test_edge_list))
    

    ## for small graphs
    # if is_directed: 
    #     edge_neg_list = list(set(permutations(np.arange(n),2))-set(edge_list))
    # else: 
    #     edge_neg_list = list(set(combinations(np.arange(n),2))-set(edge_list))
    # idx_neg = np.random.choice(len(edge_neg_list), e)
    # np.save(data_dir+"/train_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[idx_neg[:train_e],:])
    # np.save(data_dir+"/test_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[idx_neg[train_e:train_e+test_e],:])

    count_e, edge_neg_list, edge_list= 0, [], set(edge_list)
    if is_directed:
        while count_e < e:
            i,j = np.random.randint(n), np.random.randint(n)
            if i!=j and (i,j) not in edge_list and (i,j) not in edge_neg_list:
                edge_neg_list.append((i,j))
                count_e+=1
    else:
        while count_e < e:
            i,j = np.random.randint(n), np.random.randint(n)
            if i!=j and (i,j) not in edge_list and (j,i) not in edge_list \
                        and (i,j) not in edge_neg_list and (j,i) not in edge_neg_list:
                edge_neg_list.append((i,j))
                count_e+=1
    np.save(data_dir+"/train_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[:train_e,:])
    np.save(data_dir+"/test_"+str(idx)+"neg.txt.npy", np.array(edge_neg_list)[train_e:train_e+test_e,:])

    print("Used Time --- %s seconds ---" % (time.time() - start_time))
    
    return 0
