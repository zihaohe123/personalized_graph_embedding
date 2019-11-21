import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy import sparse


class AttentionWalkLayer(nn.Module):
    def __init__(self, graph, emb_dim, window_size, n_walks, beta, gamma, device):
        super(AttentionWalkLayer, self).__init__()
        n_nodes = len(graph)
        self.left_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim//2)), requires_grad=True)
        self.right_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim // 2)), requires_grad=True)
        self.attention = nn.Parameter(torch.zeros(window_size), requires_grad=True)

        self.graph = graph
        self.window_size = window_size
        self.n_walks = n_walks
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.adj_mat = None
        self.transit_mat_series = None
        self.initialize_weights()
        self.calc_transit_mat_series()
        print(self.transit_mat_series.shape)

    def initialize_weights(self):
        nn.init.uniform_(self.left_emb, -0.01, 0.01)
        nn.init.uniform_(self.right_emb, -0.01, 0.01)
        nn.init.uniform_(self.attention, -0.01, 0.01)

    def calc_transit_mat_series(self):
        print('Preparing graph...')
        adj_mat = nx.adjacency_matrix(self.graph).toarray()  # VxV
        degrees = adj_mat.sum(axis=0)  # V
        diag = np.diag(degrees)
        diag = np.linalg.inv(diag)
        transit_mat = np.dot(diag, adj_mat) + 1e-7
        transit_mat_series = [transit_mat]

        if self.window_size > 1:
            for i in range(self.window_size-1):
                transit_mat_series.append(np.dot(transit_mat_series[-1], transit_mat))

        self.adj_mat = torch.from_numpy(adj_mat).to(self.device)
        self.transit_mat_series = torch.from_numpy(np.array(transit_mat_series)).to(self.device)  # CxVxV

    def forward(self):
        attention_probs = nn.functional.softmax(self.attention, dim=0)  # C
        attention_probs = attention_probs.unsqueeze(1).unsqueeze(1).expand(self.transit_mat_series.shape)  # CxVxV
        weighted_transit_mat = self.n_walks * torch.sum(attention_probs * self.transit_mat_series, dim=0)  # VxV, E[D]
        left_dot_right = torch.mm(self.left_emb, self.right_emb.transpose(0, 1))
        loss_on_target = -weighted_transit_mat * nn.functional.logsigmoid(left_dot_right) # logsigmoid() is more numericalll stable
        loss_on_opposite = -(1-self.adj_mat) * (-left_dot_right + nn.functional.logsigmoid(left_dot_right))  # log(1-sigmoid(x)) = -x + logsigmoid(x)
        loss_on_matrices = torch.norm(loss_on_target+loss_on_opposite, p=1)
        loss_on_regularization = self.beta * torch.norm(self.attention, p=2) ** 2 \
                                 + self.gamma * (torch.norm(self.left_emb, 2) ** 2 + torch.norm(self.right_emb, 2) ** 2)
        return loss_on_matrices + loss_on_regularization

