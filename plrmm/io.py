import numpy as np
import scipy.io as spio

from .model import PLRMM, Data

def save_model(filename, model, data=None):
  dummy = {
    'alpha': model.get_alpha(),
    'sigma': model.get_sigma(),
    'pi': model.calc_pi(),
    'w': model.calc_w(),
  }
  
  if data is not None:
    pz = model.calc_pz(data)
    dummy['pz'] = pz
    dummy['assignment'] = np.argmax(pz, 1).T 

  spio.savemat(filename, dummy)

def load_model(filename):
  model = spio.loadmat(filename)

  pi = model['pi'][0]
  w = model['w']
  alpha = model['alpha']
  sigma = model['sigma']

  return PLRMM(pi, w, alpha, sigma)

def _parse_x(line):
  instance = []
  for e in line.strip().split():
    i, v = e.split(':')
    i, v = int(i), float(v)
    instance.append((i, v))
  return instance

def _parse_p(line):
  return np.array(tuple(map(int, line.strip().split())), dtype=np.int32)

def read_data(filename):
  P = []

  with open(filename, 'r') as in_file:
    num_instances, num_permutations = map(int, in_file.readline().strip().split(' '))

    num_features = 0
    instances = []
    for _ in range(num_instances):
      instance = _parse_x(in_file.readline())
      num_features = max(num_features, max(map(lambda x: x[0], instance)) + 1)
      instances.append(instance)

    for _ in range(num_permutations):
      P.append(_parse_p(in_file.readline()))

  X = np.zeros((num_instances, num_features))
  for i, x in enumerate(instances):
    for j, v in x:
      X[i, j] = v

  return Data(X, P)
