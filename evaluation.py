import numpy as np

def get_reconstructed_adj(left_embed, right_embed):
    node_num = left_embed.shape[0]

    adj = np.zeros((node_num, node_num))

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                continue
            adj[i, j] = np.dot(left_embed[i], right_embed[j])

    return adj

def get_pred_edges(adj, threshold=0.0):
    pred = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i >= j:
                continue
            if adj[i, j] > threshold:
                pred.append((i, j, adj[i, j]))
    return pred

def compute_Precision_Curve(edges, pos, max_k):

    max_k = min(max_k, len(edges))

    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)

    precision_scores = []

    delta_factors = []

    correct_edge = 0

    for i in range(max_k):
        if (sorted_edges[i][0], sorted_edges[i][1]) in pos:
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

    for edge in pred_edges:
        node_edges[edge[0]].append(edge)

    for edge in true_edges:
        node_deg[edge[0]] += 1

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

def eval_link_prediction(left_embed, right_embed, test_pos, train_pos, max_k=100):

    adj = get_reconstructed_adj(left_embed, right_embed)

    test_pred = get_pred_edges(adj)

    next_pred = [e for e in test_pred if (e[0], e[1]) not in train_pos]

    MAP = compute_MAP(test_pos, next_pred, left_embed.shape[0], max_k)

    print("MAP@{}: {}".format(max_k, MAP))
