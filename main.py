from solver import Solver
from parser import parameter_parser
from utils import tab_printer


def main():
    args = parameter_parser()
    tab_printer(args)
    solver = Solver(args)
    solver.train()
    solver.save()


if __name__ == '__main__':
    main()