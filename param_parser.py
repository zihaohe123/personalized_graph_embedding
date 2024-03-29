import argparse


def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Chameleons dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run Attention Walk.")

    parser.add_argument("--dataset", default="wiki-vote",
                        # choices=('ca-AstroPh', 'ca-HepTh', 'ppi2',
                        #          'soc-epinions', 'soc-facebook', 'wiki-vote',
                        #          'cora', 'blogcatalog', 'citeseer', 'ppi2', 'wikipedia'),
                        help="Dataset to use")
    parser.add_argument("--attention", default="global_vector",
                        choices=('constant',
                                'global_vector', 'personalized_vector',
                                'global_gamma', 'personalized_gamma',
                                'global_quadratic', 'personalized_quadratic',
                                'global_cubic', 'personalized_cubic',
                                'global_exponential', 'personalized_exponential',
                                'global_linear', 'personalized_linear',
                                'personalized_function'),
                        help="Attention method to use")
    parser.add_argument("--window-size", type=int, default=10, help="Skip-gram window size. Default is 10.")
    parser.add_argument("--emb_dim", type=int, default=128, help="Number of dimensions. Default is 64.")
    parser.add_argument("--lr", type=float, default=0.2, help="Gradient descent learning rate. Default is 0.5.")
    parser.add_argument("--shared", action='store_true', default=False, help='Shared left and right embedding')
    parser.add_argument("--normalize", default="softmax",
                        choices=('softmax', 'sum'),
                        help="Normalize method to use. Sum method assume all entries are positive")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (scaling factor) for before softmax. Default is 1.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of gradient descent iterations. Default is 1000.")
    parser.add_argument("--n-walks", type=int, default=80, help="Number of random walks. Default is 80.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Regularization parameter for adjacency matrix. Default is 0.5.")
    parser.add_argument("--gamma", type=float, default=1e-6,
                        help="Regularization parameter for embedding. Default is 1e-6.")

    parser.add_argument("--gpu", type=str, default='', help="Which GPUs to use. Default is None.")
    parser.add_argument("--output_suffix", type=str, default='default', help="output_suffix")

    return parser.parse_args()
