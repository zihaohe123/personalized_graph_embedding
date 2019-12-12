import torch
import torch.nn as nn
import scipy


class AttentionWalkLayer(nn.Module):
    def __init__(self, n_nodes, emb_dim, window_size, n_walks, beta, gamma, attention, normalize, temperature, shared):
        super(AttentionWalkLayer, self).__init__()
        self.left_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim//2)), requires_grad=True)
        if shared:
            self.right_emb = self.left_emb
        else:
            self.right_emb = nn.Parameter(torch.zeros((n_nodes, emb_dim//2)), requires_grad=True)
        self.attention_method = attention
        self.normalize_method = normalize
        self.attention = None
        self.q = None
        self.k = None
        self.theta = None
        self.a = None
        self.b = None
        self.c = None
        self.temperature = temperature

        if attention == 'constant':
            self.attention = nn.Parameter(torch.ones(window_size), requires_grad=False)
        elif attention == 'global_vector':
            self.attention = nn.Parameter(torch.ones(window_size), requires_grad=True)
        elif attention == 'global_exponential':
            self.q = nn.Parameter(torch.ones(1), requires_grad=True)
        elif attention == 'global_gamma':
            self.k = nn.Parameter(torch.ones(1), requires_grad=True)
            self.theta = nn.Parameter(torch.ones(1), requires_grad=True)
            # theta_k = torch.pow(self.theta, self.k)
            # gamma_k = torch.from_numpy(scipy.special.gamma(self.k.detach().numpy()))
            # self.coeff = (theta_k/gamma_k).detach()
        elif attention == 'global_quadratic':
            self.a = nn.Parameter(torch.tensor([-1.]), requires_grad=True)
            self.b = nn.Parameter(torch.ones(1), requires_grad=True)
            self.c = nn.Parameter(torch.ones(1), requires_grad=True)
        elif attention == 'personalized_vector':
            self.attention = nn.Parameter(torch.ones((window_size, n_nodes)), requires_grad=True)
        elif attention == 'personalized_exponential':
            self.q = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
        elif attention == 'personalized_linear':
            self.q = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
        elif attention == 'personalized_gamma':
            self.k = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
            self.theta = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
        elif attention == 'personalized_quadratic':
            self.a = nn.Parameter(torch.tensor([-1.]*n_nodes), requires_grad=True)
            self.b = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
            self.c = nn.Parameter(torch.ones(n_nodes), requires_grad=True)
        elif attention == 'personalized_function':
            self.linear = nn.Linear(emb_dim//2, window_size)
            nn.init.zeros_(self.linear.bias)
        else:
            print('Unexpected attention method')
            exit()

        self.n_nodes = n_nodes
        # self.transit_mat_series = transit_mat_series
        self.window_size = window_size
        self.n_walks = n_walks
        self.beta = beta
        self.gamma = gamma
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.left_emb, -0.1, 0.1)
        nn.init.uniform_(self.right_emb, -0.1, 0.1)
        # nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self, transit_mat):
        if self.attention_method in ('global_exponential', 'personalized_exponential'):
            q_abs = torch.abs(self.q)
            mults = []
            for i in range(self.window_size):
                mults.append(0.99 * (q_abs ** i) + 0.01)
            self.attention = torch.stack(mults)
        elif self.attention_method == 'personalized_linear':
            q_neg = -1 * self.q
            mults = []
            for i in range(self.window_size):
                mults.append(q_neg * i)
            self.attention = torch.stack(mults)
        elif self.attention_method in ['global_gamma', 'personalized_gamma']:
            mults = []
            # k = 10*torch.sigmoid(self.k)
            # theta = 2*torch.sigmoid(self.theta)
            for i in range(1, self.window_size+1):
                mults.append(torch.pow(i, self.k-1)*torch.exp(-self.theta*i))
            self.attention = torch.stack(mults)
        elif self.attention_method in ['global_quadratic', 'personalized_quadratic']:
            mults = []
            for i in range(1, self.window_size+1):
                mults.append(self.a*i**2 + self.b*i + self.c)
            self.attention = torch.stack(mults)
        elif self.attention_method == 'personalized_function':
            self.attention = torch.t(self.linear(self.left_emb))  # n_nodes*window_size --> window_size*n_nodes

        if self.normalize_method == 'softmax':
            attention_probs = nn.functional.softmax(self.attention * self.temperature, dim=0)  # C
        elif self.normalize_method == 'sum':
            attention_probs = self.attention / (torch.sum(self.attention, dim=0))
        else:
            print('Unexpected normalize method')
            exit()

        transit_mat_power_n = torch.diag(torch.ones(self.n_nodes, dtype=torch.float, device=transit_mat.device))
        weighted_transit_mat = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.float, device=transit_mat.device)  # VxV
        for i in range(self.window_size):
            transit_mat_power_n = torch.mm(transit_mat_power_n, transit_mat)
            weighted_transit_mat += attention_probs[i] * transit_mat_power_n
        weighted_transit_mat *= self.n_walks * self.n_nodes

        left_dot_right = torch.mm(self.left_emb, self.right_emb.transpose(0, 1))

        loss_on_target = -weighted_transit_mat * nn.functional.logsigmoid(left_dot_right) # logsigmoid() is more numerically stable
        loss_on_opposite = -(1-transit_mat) * (-left_dot_right + nn.functional.logsigmoid(left_dot_right))  # log(1-sigmoid(x)) = -x + logsigmoid(x)
        loss_on_matrices = torch.mean(torch.abs(loss_on_target+loss_on_opposite))
        loss_on_regularization = self.beta * torch.mean(self.attention**2) \
                                 + self.gamma * (torch.mean(self.left_emb**2) + torch.mean(self.right_emb**2))

        if self.q is not None:
            loss_on_regularization += self.gamma * torch.mean(self.q)**2
        if self.k is not None:
            loss_on_regularization += self.gamma * torch.mean(self.k)**2
        if self.theta is not None:
            loss_on_regularization += self.gamma * torch.mean(self.theta)**2
        if self.a is not None:
            loss_on_regularization += self.gamma * torch.mean(self.a)**2
        if self.b is not None:
            loss_on_regularization += self.gamma * torch.mean(self.b)**2
        if self.c is not None:
            loss_on_regularization += self.gamma * torch.mean(self.c)**2

        return loss_on_matrices + loss_on_regularization
