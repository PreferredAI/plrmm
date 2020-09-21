import argparse
import sys

import scipy.io as spio
from plrmm.io import load_model, read_data


def save_ranks(filename, ranks):
  with open(filename, 'w') as out_file:
    for rank in ranks:
      print(' '.join(map(str, rank)), file=out_file)

def main(args):
  argparser = argparse.ArgumentParser()
  argparser.add_argument('model', help="model '*.mat' file")
  argparser.add_argument('examples', help='examples to be ranked')
  argparser.add_argument('assignment', help='examples "info" file')
  argparser.add_argument('prediction', help='prediction output')

  args = argparser.parse_args(args)

  model = load_model(args.model)
  examples = read_data(args.examples)
  pz = spio.loadmat(args.assignment)['assignment'][0]

  ranks = model.calc_ranks(examples, pz)
  save_ranks(args.prediction, ranks)

if __name__ == '__main__':
  main(sys.argv[1:])
