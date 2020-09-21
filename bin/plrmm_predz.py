import argparse
import sys

import numpy as np
import scipy.io as spio
from plrmm.io import load_model, read_data


def main(args):
  argparser = argparse.ArgumentParser()
  argparser.add_argument('model', help="model '*.mat' file")
  argparser.add_argument('examples', help='examples to be classified')
  argparser.add_argument('prediction', help='prediction output')

  args = argparser.parse_args(args)

  model = load_model(args.model)
  examples = read_data(args.examples)
  pz = model.calc_pz(examples)
  
  spio.savemat(args.prediction, {'pz': pz, 'assignment': np.argmax(pz, 1).T})

if __name__ == '__main__':
  main(sys.argv[1:])
