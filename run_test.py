import argparse
from DebyeCalculatorGPU import DebyeCalculatorGPU

def test(
    args
):
    DB = DebyeCalculatorGPU()
    print(DB.ff_coef['H'])
    r, pdf = DB.gr(args['xyz'])
    print(pdf)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xyz', type=str)
    args = vars(argparser.parse_args())
    test(args)


