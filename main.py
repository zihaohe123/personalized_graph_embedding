from solver import Solver
from param_parser import parameter_parser
from utils import tab_printer
import os


def main():
    args = parameter_parser()
    tab_printer(args)
    if not os.path.exists('datasets'):
        print('Downloading datasets...')
        os.system('curl http://sami.haija.org/graph/datasets.tgz > datasets.tgz')
        os.system('tar zxvf datasets.tgz')
    solver = Solver(args)
    solver.train()
    solver.save()


if __name__ == '__main__':
    main()
