import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle
import os
from attentionwalk import AttentionWalkLayer
from evaluation import *
import pdb
import networkx as nx
from sklearn.model_selection import train_test_split


class Solver:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.optimizer = None
        self.device = 'cpu'
        self.num_workers = 4

        self.is_directed = False
        self.num_nodes = 0
        self.test_neg_arr = None
        self.test_pos_arr = None
        self.train_pos_arr = None
        self.train_neg_arr = None
        self.adj_mat = None
        self.transit_mat = None
        self.eval_metrics = None
        self.node_labels = None
        self.node_list_map = None
        self.task = 'lp'    # link prediction (lp) or node classification (nc)
        self.eval_metrics = {}

        description = 'wz_{}+emb_{}+lr_{}{}{}{}{}{}{}'.format(
            self.args.window_size,
            self.args.emb_dim,
            self.args.lr,
            '+shared' if self.args.shared else '',
            '+normalize_sum' if self.args.normalize == 'sum' else '',
            '+temperature_{}'.format(self.args.temperature) if self.args.temperature != 1.0 else '',
            '+nwalks_{}'.format(self.args.n_walks) if self.args.n_walks != 80 else '',
            '+beta_{}'.format(self.args.beta) if self.args.beta != 0.5 else '',
            '+gamma_{}'.format(self.args.gamma) if args.gamma != 1e-5 else '',
        )

        output_path = os.path.join('output', self.args.dataset, self.args.attention, description, self.args.output_suffix)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

        self.prepare_graph()
        self.init_training()

    def prepare_graph(self):
        print('Loading graph....')

        dataset_dir = os.path.join('datasets', self.args.dataset)

        if 'nc' in self.args.dataset:
            self.task = 'nc'    # node classification

        test_neg_file = os.path.join(dataset_dir, 'test.directed.neg.txt.npy')
        if os.path.exists(test_neg_file):
            self.is_directed = True
        else:
            self.is_directed = False
            test_neg_file = os.path.join(dataset_dir, 'test.neg.txt.npy')
        test_neg_arr = np.load(open(test_neg_file, 'rb'))

        test_pos_file = os.path.join(dataset_dir, 'test.txt.npy')
        test_pos_arr = np.load(open(test_pos_file, 'rb'))

        train_pos_file = os.path.join(dataset_dir, 'train.txt.npy')
        train_neg_file = os.path.join(dataset_dir, 'train.neg.txt.npy')
        train_pos_arr = np.load(open(train_pos_file, 'rb'))
        train_neg_arr = np.load(open(train_neg_file, 'rb'))

        index_file = os.path.join(dataset_dir, 'index.pkl')
        if os.path.exists(index_file):      # sami's dataset, must be lp task
            index = pickle.load(open(index_file, 'rb'))
            self.num_nodes = len(index['index'])
        else:
            G = nx.read_gpickle(os.path.join(dataset_dir, 'train.gpickle'))
            self.num_nodes = len(G.nodes())
            if self.task == 'nc':
                label_path = os.path.join(dataset_dir, 'node_labels.pickle')
                self.node_labels = pickle.load(open(label_path, 'rb')).toarray()
                label_map_path = os.path.join(dataset_dir, 'nodelistmap.pickle')
                self.node_list_map = pickle.load(open(label_map_path, 'rb'))

        self.test_neg_arr = test_neg_arr
        self.test_pos_arr = test_pos_arr
        self.train_pos_arr = train_pos_arr
        self.train_neg_arr = train_neg_arr

        adj_mat = np.zeros((self.num_nodes, self.num_nodes), dtype='float32')
        train_edges = np.load(open(os.path.join(dataset_dir, 'train.txt.npy'), 'rb'))
        adj_mat[train_edges[:, 0], train_edges[:, 1]] = 1.0
        if not self.is_directed:
            adj_mat[train_edges[:, 1], train_edges[:, 0]] = 1.0

        print('#Nodes', self.num_nodes)
        print('#Edges', len(train_edges))
        print('Is_directed', self.is_directed)
        print('Preparing graph...')

        # how to use GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = max([4 * torch.cuda.device_count(), 4])

        import copy
        transit_mat = copy.deepcopy(adj_mat.T)
        degree = transit_mat.sum(axis=0)
        transit_mat = transit_mat / (degree + 1e-7)
        self.adj_mat = torch.from_numpy(adj_mat)
        self.transit_mat = torch.from_numpy(transit_mat)

    def init_training(self):
        print('Initializing training....')

        self.model = AttentionWalkLayer(self.num_nodes, self.args.emb_dim, self.args.window_size,
                                        self.args.n_walks, self.args.beta, self.args.gamma, self.args.attention,
                                        self.args.normalize, self.args.temperature, self.args.shared)

        if self.device == 'cuda':
            device_count = torch.cuda.device_count()
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30)

    def train(self):
        print("Training the model....\n")
        best_train_loss = 9999999999999999
        best_train_loss_epoch = 0
        self.model.train()
        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            self.adj_mat = self.adj_mat.to(self.device)
            self.transit_mat = self.transit_mat.to(self.device)
            loss = self.model(self.adj_mat, self.transit_mat)
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                best_train_loss_epoch = epoch
                self.save_ckp()

            if epoch % 10 == 0 or epoch+1 == self.args.epochs:
                if self.task == 'lp':
                    print('Epoch: {:0>3d}/{}, '
                          'Loss: {:.4f}, '
                          'Best Loss: {:.4f}, '
                          'Epoch at Best Train: {:0>3d}'.format(epoch+1, self.args.epochs,
                                                                loss,
                                                                best_train_loss,
                                                                best_train_loss_epoch+1
                                                                ))
                else:
                    print('Epoch: {:0>3d}/{}, '
                          'Loss: {:.4f}, '
                          'Best Loss: {:.4f}, '
                          'Epoch at Best Train: {:0>3d}'.format(epoch+1, self.args.epochs,
                                                                loss,
                                                                best_train_loss,
                                                                best_train_loss_epoch+1
                                                                ))

                if epoch - best_train_loss_epoch >= 50:
                    print('The model seems to become overfitting...')
                    break

        print('Training Finished. Evaluating....')

    def node_classification_eval(self, test_ratio=0.3):
        # micro, macro = 0, 0
        #
        # if self.node_labels is None:
        #     print("Node labels are not provided...")
        #     return micro, macro

        if self.args.shared:
            embeds = self.model.left_emb.detach().to('cpu').numpy()
        else:
            embeds = torch.cat((self.model.left_emb, self.model.right_emb), dim=1).detach().to('cpu').numpy()

        temp_map = {v: k for k, v in self.node_list_map.items()}
        labels = np.array([self.node_labels[temp_map[i]] for i in range(self.num_nodes)])

        micros, macros = [], []

        for seed in range(0, 25, 5):
            X_tr, X_te, y_tr, y_te = train_test_split(embeds,
                                                      labels,
                                                      test_size=test_ratio,
                                                      random_state=seed)

            micro, macro = eval_node_classification(X_tr, y_tr, X_te, y_te)
            micros.append(micro)
            macros.append(macro)

        return sum(micros) / len(micros), sum(macros) / len(macros)

    def link_prediction_eval(self):
        """Calls sess.run(g) and computes AUC metric for test and train."""

        scores = torch.mm(self.model.left_emb, self.model.right_emb.transpose(0, 1)).detach().to('cpu').numpy()

        # Compute train auc:
        train_pos_prods = scores[self.train_pos_arr[:, 0], self.train_pos_arr[:, 1]]
        train_neg_prods = scores[self.train_neg_arr[:, 0], self.train_neg_arr[:, 1]]
        train_y = [0] * len(train_neg_prods) + [1] * len(train_pos_prods)
        train_y_pred = np.concatenate([train_neg_prods, train_pos_prods], 0)
        train_auc = metrics.roc_auc_score(train_y, train_y_pred)

        # Compute test auc:
        test_pos_prods = scores[self.test_pos_arr[:, 0], self.test_pos_arr[:, 1]]
        test_neg_prods = scores[self.test_neg_arr[:, 0], self.test_neg_arr[:, 1]]
        test_y = [0] * len(test_neg_prods) + [1] * len(test_pos_prods)
        test_y_pred = np.concatenate([test_neg_prods, test_pos_prods], 0)
        test_auc = metrics.roc_auc_score(test_y, test_y_pred)

        test_map = eval_link_prediction(self.model.left_emb,
                                        self.model.right_emb,
                                        self.test_pos_arr,
                                        self.train_pos_arr,
                                        is_directed=self.is_directed)
        return train_auc, test_auc, test_map

    def save_embedding(self):
        print("Saving the embedding....")
        left_emb = self.model.left_emb.detach().to('cpu').numpy()
        right_emb = self.model.right_emb.detach().to('cpu').numpy()
        if not self.args.shared:
            embedding = np.concatenate([left_emb, right_emb], axis=1)
            columns = ["x_" + str(x) for x in range(self.args.emb_dim)]
        else:
            embedding = np.concatenate([left_emb], axis=1)
            columns = ["x_" + str(x) for x in range(self.args.emb_dim//2)]

        embedding = pd.DataFrame(embedding, columns=columns)
        embedding_path = os.path.join(self.output_path, 'embedding.csv')
        embedding.to_csv(embedding_path, index=None)

    def save_attention(self):
        print("Saving the attention....")
        if self.args.attention in ['global_exponential', 'personalized_exponential']:
            q = self.model.q.detach().to('cpu').numpy().reshape(-1, 1)
            q = pd.DataFrame(q)
            q_path = os.path.join(self.output_path, 'q.csv')
            q.to_csv(q_path, index=None)
        elif self.args.attention in ['global_gamma', 'personalized_gamma']:
            k = self.model.k.detach().to('cpu').numpy().reshape(-1, 1)
            theta = self.model.theta.detach().to('cpu').numpy().reshape(-1, 1)
            data = np.concatenate((k, theta), axis=1)
            df = pd.DataFrame(data, columns=['k', 'theta'])
            path = os.path.join(self.output_path, 'k_theta.csv')
            df.to_csv(path, index=None)
        elif self.args.attention in ['global_quadratic', 'personalized_quadratic']:
            a = self.model.a.detach().to('cpu').numpy().reshape(-1, 1)
            b = self.model.b.detach().to('cpu').numpy().reshape(-1, 1)
            c = self.model.c.detach().to('cpu').numpy().reshape(-1, 1)
            data = np.concatenate((a, b, c), axis=1)
            df = pd.DataFrame(data, columns=['a', 'b', 'c'])
            path = os.path.join(self.output_path, 'a_b_c.csv'.format(self.args.dataset, self.args.attention))
            df.to_csv(path, index=None)
        elif self.args.attention in ['global_cubic', 'personalized_cubic']:
            a = self.model.a.detach().to('cpu').numpy().reshape(-1, 1)
            b = self.model.b.detach().to('cpu').numpy().reshape(-1, 1)
            c = self.model.c.detach().to('cpu').numpy().reshape(-1, 1)
            d = self.model.d.detach().to('cpu').numpy().reshape(-1, 1)
            data = np.concatenate((a, b, c, d), axis=1)
            df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'])
            path = os.path.join(self.output_path, 'a_b_c_d.csv')
            df.to_csv(path, index=None)
        if self.args.normalize == 'softmax':
            attention = nn.functional.softmax(self.model.attention * self.model.temperature, dim=0)     # window_size*n_nodes
        else:
            attention = self.model.attention / (torch.sum(self.model.attention, dim=0))
        if attention.ndim > 1:
            attention = attention.transpose(0, 1)
        attention = attention.detach().to('cpu').numpy().reshape(-1, self.args.window_size)
        columns = ["x_" + str(x) for x in range(self.args.window_size)]
        df = pd.DataFrame(attention, columns=columns)
        attention_path = os.path.join(self.output_path, 'attention.csv')
        df.to_csv(attention_path, index=None)

    def save_results(self):
        print("Saving the results....")

        train_auc, test_auc, test_map = self.link_prediction_eval() if self.task == 'lp' else (0, 0, 0)
        nc_micro, nc_macro = self.node_classification_eval() if self.task == 'nc' else (0, 0)
        if self.task == 'lp':
            results = 'Test AUC: {:.4f}, Test mAP: {:.4f}, \n'.format(test_auc, test_map)
        else:
            results = 'Test NC Micro-F1: {:.4f}, Test NC Macro-F1: {:.4f},\n'.format(nc_micro, nc_macro)
        print(results)

        path = os.path.join(self.output_path, 'results.txt')
        with open(path, mode='w') as f:
            f.write(results)

    def save_ckp(self):
        ckp = {'state_dict': self.model.state_dict()}
        torch.save(ckp, os.path.join(self.output_path, 'model.pth'))

    def load_ckp(self):
        print('Loading checkpoint....')
        ckp = torch.load(os.path.join(self.output_path, 'model.pth'), map_location=self.device)
        self.model.load_state_dict(ckp['state_dict'])

    def save(self):
        self.load_ckp()
        self.model.update_attention()
        self.save_results()
        self.save_embedding()
        self.save_attention()
