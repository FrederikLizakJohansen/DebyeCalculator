import argparse
from debye_calculator import DebyeCalculator

def test(
    args
):
    DB = DebyeCalculator(rthres=1)
    print(DB.FORM_FACTOR_COEF['H'])
    r, pdf = DB.gr(args['xyz'])
    print(pdf)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xyz', type=str, required=True)
    args = vars(argparser.parse_args())
    test(args)


