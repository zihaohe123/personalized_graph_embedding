import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
import os
from attentionwalk import AttentionWalkLayer


class Solver:
    def __init__(self, args):
        self.args = args
        self.graph = None
        self.model = None
        self.optimizer = None
        self.device = 'cpu'
        self.num_workers = 4
        self.read_graph()
        self.init_training()

    def read_graph(self):
        print('Loading graph....')
        edgelist = pd.read_csv(self.args.graph_path).values.tolist()
        graph = nx.from_edgelist(edgelist)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        self.graph = graph

    def init_training(self):
        print('Initializing training....')

        # how to use GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = max([4 * torch.cuda.device_count(), 4])

        self.model = AttentionWalkLayer(self.graph, self.args.emb_dim, self.args.window_size,
                                        self.args.n_walks, self.args.beta, self.args.gamma, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        if self.device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        self.model.to(self.device)

    def train(self):
        print("Training the model....")
        self.model.train()
        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 1000 == 0 or epoch+1 == self.args.epochs:
                print('Epoch: {:0>5d}/{:0>5d}, Loss: {:.2f}'.format(epoch+1, self.args.epochs, loss))

    def save_embedding(self):
        print("Saving the embedding....")
        left_emb = self.model.left_emb.detach().to('cpu').numpy()
        right_emb = self.model.right_emb.to('cpu').detach().numpy()
        indices = np.arange(len(self.graph)).reshape(-1, 1)
        embedding = np.concatenate([indices, left_emb, right_emb], axis=1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.emb_dim)]
        embedding = pd.DataFrame(embedding, columns=columns)
        embedding.to_csv(self.args.embedding_path, index=None)

    def save_attention(self):
        print("Saving the attention....")
        attention = nn.functional.softmax(self.model.attention, dim=0).detach().to('cpu').numpy().reshape(-1, 1)
        indices = np.arange(self.args.window_size).reshape(-1, 1)
        attention = np.concatenate([indices, attention], axis=1)
        attention = pd.DataFrame(attention, columns=['Order', 'Weight'])
        attention.to_csv(self.args.attention_path, index=None)

    def save(self):
        self.save_embedding()
        self.save_attention()