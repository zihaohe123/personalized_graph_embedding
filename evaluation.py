from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
import numpy as np
import torch

def get_reconstructed_adj(left_embed, right_embed, is_directed=False):
    n = left_embed.shape[0]

    if is_directed:
        di = np.diag_indices(n)
    else:
        di = np.tril_indices(n)

    adj = torch.mm(torch.tensor(left_embed), torch.tensor(right_embed).transpose(0, 1))

    adj[di[0], di[1]] = 0

    return adj

def get_pred_edges(adj, threshold=0.0):

    mask = torch.ones(adj.shape[0]).float()

    mask = 1 - mask.diag()

    ind = torch.nonzero((adj > threshold).float() *mask).to("cpu").numpy()

    weights = adj[ind[:, 0], ind[:, 1]].to("cpu").numpy()

    return list(zip(ind, weights))

def compute_Precision_Curve(edges, pos, max_k):

    max_k = min(max_k, len(edges))

    sorted_edges = sorted(edges, key=lambda x: x[1], reverse=True)

    precision_scores = []

    delta_factors = []

    correct_edge = 0

    for i in range(max_k):
        edge = sorted_edges[i][0]
        if (edge[0], edge[1]) in pos:
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)

        precision_scores.append(1.0 * correct_edge / (i + 1))

    return precision_scores, delta_factors


def compute_MAP(true_edges, pred_edges, node_num, max_k):
    node_edges = []
    node_deg = []

    for i in range(node_num):
        node_edges.append([])
        node_deg.append(0)

    for (e, w) in pred_edges:
        node_edges[e[0]].append((e, w))

    for e in true_edges:
        node_deg[e[0]] += 1

    node_AP = [0.0] * node_num

    count = 0

    for i in range(node_num):
        if node_deg[i] == 0:
            continue
        count += 1

        precision_scores, delta_factors = compute_Precision_Curve(node_edges[i], true_edges, max_k)

        precision_rectified = [p * d for p,d in zip(precision_scores, delta_factors)]

        if(sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))

    return sum(node_AP) / count

def eval_link_prediction(left_embed, right_embed, test_pos, train_pos,
                         max_k=100, is_directed=False):

    left_embed = left_embed.detach().to('cpu').numpy()

    right_embed = right_embed.detach().to('cpu').numpy()

    test_pos_set, train_pos_set = set(), set()

    for edge in test_pos:
        test_pos_set.add((edge[0], edge[1]))

    for edge in train_pos:
        train_pos_set.add((edge[0], edge[1]))

    adj = get_reconstructed_adj(left_embed, right_embed, is_directed)

    test_pred = get_pred_edges(adj)

    next_pred = [e for e in test_pred if (e[0][0], e[0][1]) not in train_pos_set]

    MAP = compute_MAP(test_pos_set, next_pred, left_embed.shape[0], max_k)

    return MAP

class TopKRanker(oneVr):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)

        probs = np.asarray(super(TopKRanker, self).predict_proba(X))

        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))

        for i, k in enumerate(top_k_list):

            probs_ = probs[i, :]

            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()

            for label in labels:
                prediction[i, label] = 1

        return prediction


def eval_node_classification(X_train, Y_train, X_test, Y_test):

    # y_train = (n_sample, n_classes)

    top_k_list = list(Y_test.sum(axis=1))

    classif2 = TopKRanker(lr())

    classif2.fit(X_train, Y_train)

    prediction = classif2.predict(X_test, top_k_list)

    micro = f1_score(Y_test, prediction, average='micro')

    macro = f1_score(Y_test, prediction, average='macro')

    return (micro, macro)
