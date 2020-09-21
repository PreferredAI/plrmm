import numpy as np
from scipy.optimize import minimize

from .plfunc import plackett_luce_pdf
from .plfunc import plackett_luce_log_pdf
from .plfunc import plackett_luce_log_pdf_diff

class Data:
  
  def __init__(self, X, P):
    self.X = X
    self.P = P

  def __str__(self):
    return 'X:\n' + str(self.X) + '\n\nP:\n' + str(self.P)

class PLRMM:

  def __init__(self, pi, w, alpha, sigma):
    self._pi = pi
    self._w = w
    self._k = len(pi)
    self._d = w.shape[1]
    self._alpha = alpha
    self._sigma = sigma
    self._sigma2 = sigma * sigma

  def get_sigma(self):
    return self._sigma
    
  def get_alpha(self):
    return self._alpha

  def calc_w(self):
    return self._w

  def calc_pi(self):
    return self._pi

  def calc_pz(self, data, pz=None):
    m = len(data.P)
    k = self._k

    pi = self._pi
    w = self._w

    if pz is None:
      pz = np.zeros((m, self._k))

    xw = np.dot(data.X, w.T)
    for i in range(m):
      xwi = xw[data.P[i]].T
      for j in range(k):
        pz[i, j] = pi[j] * plackett_luce_pdf(xwi[j])
      pz[i] /= np.sum(pz[i])

    return pz

  def calc_ranks(self, data, zs):
    w = self._w
    m = len(zs)

    ranks = []
    for i, z in enumerate(zs):
      xi = data.X[data.P[i]]
      ranks.append(np.dot(xi, w[z]))

    return ranks

  def log_likelihood(self, data, pz=None):
    m = len(data.P)
    k = self._k

    alpha = self._alpha
    sigma2 = self._sigma2

    log_pi = np.log(self._pi)
    w = self._w

    if pz is None:
      pz = self.calc_pz(data)

    lb = 0.0
    if alpha != 1.0:
      lb += (alpha - 1.0) * np.sum(log_pi)

    wf = w.reshape(-1)
    lb += -0.5 * np.dot(wf, wf) / sigma2

    xw = np.dot(data.X, w.T)
    for i in range(m):
      xwi = xw[data.P[i]].T
      for j in range(k):
        lb += pz[i, j] * (log_pi[j] + plackett_luce_log_pdf(xwi[j]))

    pzf = pz.reshape(-1)
    lb -= np.dot(pzf, np.log(pzf))

    return lb

  def em(self, data, num_iter=50, tol=1e-6):
    m = len(data.P)
    k = self._k
    d = self._d
    n = data.X.shape[0]

    alpha = self._alpha

    pz = np.zeros((m, k))

    plb = np.NINF
    for _ in range(num_iter):
      # Expectation Step:

      pz = self.calc_pz(data, pz)

      # Calculating Lower Bound:

      lb = self.log_likelihood(data, pz)

      if np.isfinite(lb - plb):
        print('LB = {0} (delta = {1})'.format(lb, lb - plb))
        if lb - plb < tol:
          print('Optimization is Done (tolerance criteria)')
          break
      else:
        print('LB = {0}'.format(lb))

      if plb > lb:
        print('WARNING: the previous LB is better than the new estimate', file=sys.stderr)

      plb = lb

      # Maximization Step:

      pi = np.sum(pz, 0) + alpha - 1.0
      pi /= np.sum(pi)
      self._pi = pi

      def fun(w):
        wf = w.reshape(-1)
        wv = w.reshape((k, d))

        result = -0.5 * np.dot(wf, wf) / self._sigma2

        xw = np.dot(data.X, wv.T)
        for i in range(m):
          xwi = xw[data.P[i]].T
          for j in range(k):
            result += pz[i, j] * plackett_luce_log_pdf(xwi[j])

        return -result

      def jac(w):
        wv = w.reshape((k, d))
        wxs = np.dot(wv, data.X.T)

        result = - np.copy(wv) / self._sigma2
        for j in range(k):
          rj = result[j]
          wxj = wxs[j]
          for i in range(m):
            p = data.P[i]
            xi = data.X[p]
            wxij = wxj[p]
            rj += pz[i, j] * plackett_luce_log_pdf_diff(xi, wxij)

        return -result.flatten()

      res = minimize(fun, self._w, jac=jac, method='L-BFGS-B')
      self._w = res.x.reshape((k, d))
      
    else:
      print('Optimization is Done (number of iterations exceeded)')
