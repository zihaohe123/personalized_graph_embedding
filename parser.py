import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Chameleons dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run Attention Walk.")
    parser.add_argument("--graph-path", default="input/chameleon_edges.csv", help="Edge list csv.")
    parser.add_argument("--embedding-path", default="output/chameleon_AW_embedding.csv", help="Target embedding csv.")
    parser.add_argument("--attention-path", default="output/chameleon_AW_attention.csv", help="Attention vector csv.")
    parser.add_argument("--emb_dim", type=int, default=128, help="Number of dimensions. Default is 128.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of gradient descent iterations. Default is 200.")
    parser.add_argument("--window-size", type=int, default=10, help="Skip-gram window size. Default is 10.")
    parser.add_argument("--n-walks", type=int, default=80, help="Number of random walks. Default is 80.")
    parser.add_argument("--beta", type=float, default=0.5, help="Regularization parameter. Default is 0.5.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Regularization parameter. Default is 0.5.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Gradient descent learning rate. Default is 0.01.")
    parser.add_argument("--gpu", type=str, default='', help="Which GPUs to use. Default is None.")

    return parser.parse_args()
