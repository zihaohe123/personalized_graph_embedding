import torch
import torch.nn as nn
import pdb

class AttentionWalkLayer(nn.Module):
    def __init__(self, n_nodes, transit_mat_series, emb_dim, window_size, n_walks, beta, gamma, attention, device):
        super(AttentionWalkLayer, self).__init__()
        self.left_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim//2)), requires_grad=True)
        self.right_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim//2)), requires_grad=True)
        if attention == 'constant':
            self.attention = torch.ones(window_size).to(device)
        elif attention == 'global_vector':
            self.attention = nn.Parameter(torch.ones(window_size), requires_grad=True)
        elif attention == 'global_exponential':
            self.q = nn.Parameter(torch.ones(1), requires_grad=True)
            q_sigmoid = torch.sigmoid(self.q)
            mults = []
            for i in range(window_size):
                mults.append(0.99 * (q_sigmoid ** i) + 0.01)
            self.attention = torch.stack(mults).to(device)
        elif attention == 'personalized_vector':
            self.attention = nn.Parameter(torch.ones((window_size, transit_mat_series.shape[1])), requires_grad=True)
        elif attention == 'personalized_exponential':
            self.q = nn.Parameter(torch.ones(transit_mat_series.shape[1]), requires_grad=True)
            q_sigmoid = torch.sigmoid(self.q)
            mults = []
            for i in range(window_size):
                mults.append(0.99 * (q_sigmoid ** i) + 0.01)
            self.attention = torch.stack(mults).to(device)
        elif attention == 'personalized_linear':
            self.q = -1 * nn.Parameter(torch.ones(transit_mat_series.shape[1]), requires_grad=True)
            mults = []
            for i in range(window_size):
                mults.append(self.q * i)
            self.attention = torch.stack(mults).to(device)
        elif attention == 'personalized_function':
            self.weight = nn.Parameter(torch.Tensor(emb_dim//2, window_size), requires_grad=True)
            self.attention = torch.t(torch.matmul(self.left_emb, self.weight)).to(device)
        else:
            print('Unexpected attention method')
            exit()

        self.n_nodes = n_nodes
        self.transit_mat_series = transit_mat_series
        self.window_size = window_size
        self.n_walks = n_walks
        self.beta = beta
        self.gamma = gamma
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.left_emb, -0.1, 0.1)
        nn.init.uniform_(self.right_emb, -0.1, 0.1)
        # nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self):
        attention_probs = nn.functional.softmax(self.attention, dim=0)  # C
        attention_probs = attention_probs.unsqueeze(1).unsqueeze(1)  # Cx1x1
        weighted_transit_mat = self.n_walks * self.n_nodes * torch.sum(attention_probs * self.transit_mat_series, dim=0)  # VxV, E[D]
        left_dot_right = torch.mm(self.left_emb, self.right_emb.transpose(0, 1))

        loss_on_target = -weighted_transit_mat * nn.functional.logsigmoid(left_dot_right) # logsigmoid() is more numerically stable
        loss_on_opposite = -(1-self.transit_mat_series[0]) * (-left_dot_right + nn.functional.logsigmoid(left_dot_right))  # log(1-sigmoid(x)) = -x + logsigmoid(x)
        loss_on_matrices = torch.mean(torch.abs(loss_on_target+loss_on_opposite))

        loss_on_regularization = self.beta * torch.mean(self.attention**2) \
                                 + self.gamma * (torch.mean(self.left_emb**2) + torch.mean(self.right_emb**2))

        return loss_on_matrices + loss_on_regularization
