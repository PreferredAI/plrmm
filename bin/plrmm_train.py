import argparse
import sys

import numpy as np
import scipy.io as spio
from plrmm.io import read_data, save_model
from plrmm.model import PLRMM


def train_plrm(data, params):
  k = params['k']
  d = params['d']
  
  alpha = params['alpha']
  sigma = params['sigma']

  np.random.seed(params['seed'])  

  w = np.random.normal(scale=sigma, size=(k, d)) 
  pi = np.ones(k) / k

  model = PLRMM(pi, w, alpha, sigma)

  model.em(data, num_iter=params['iter'], tol=params['tol'])

  return model

def main(args):
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-k', help='number of latent ranking functions', type=int, default=2)
  argparser.add_argument('-s', help='Gaussian regularization (\\sigma)', type=float, default=1.0)
  argparser.add_argument('-a', help='Dirichlet prior for the topic distribution', type=float, default=2.0)
  argparser.add_argument('-i', help='number of iterations (50 is default)', type=int, default=50)
  argparser.add_argument('-t', help='tolerance (1e-6 is default)', type=float, default=1e-6)
  argparser.add_argument('-r', help='random seed (if seed is -1, NumPy default initialization is used)', type=int, default=1)
  argparser.add_argument('input', help='input file')
  argparser.add_argument('model', help='model file')

  args = argparser.parse_args(args)
  
  data = read_data(args.input)

  params = {
    'k': args.k,
    'sigma': args.s,
    'alpha': args.a,
    'n': data.X.shape[0],
    'd': data.X.shape[1],
    'm': len(data.P),
    'seed': None if args.r == -1 else args.r,
    'iter': args.i,
    'tol': args.t,
  }

  model = train_plrm(data, params)
  
  save_model(args.model, model, data)

if __name__ == '__main__':
  main(sys.argv[1:])
