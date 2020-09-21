import numpy

from libc.math cimport exp, log
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef plackett_luce_log_pdf(np.ndarray[double, ndim=1] v):
  cdef int i, j
  cdef int n = v.shape[0]
  cdef np.ndarray[double, ndim=1] max_v = v.copy()
  cdef double log_pdf = 0.0
  cdef double z = 0.0
  cdef exp_sum = 0.0

  for i in range(n-2,-1,-1): 
    if max_v[i] < max_v[i+1]:
      max_v[i] = max_v[i+1]

  for i in range(n-1,-1,-1):
    z = max_v[i]
    log_pdf += v[i] - z
    exp_sum = 0.0
    for j in range(i, n):
      exp_sum += exp(v[j] - z)
    log_pdf -= log(exp_sum)

  return log_pdf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def plackett_luce_log_pdf_diff(np.ndarray Xs, np.ndarray v):
  cdef int i, j
  cdef int n = v.shape[0]
  cdef int d = Xs.shape[1]
  cdef double[:] cs = numpy.exp(v - v.max())
  cdef np.ndarray buf = numpy.zeros((d,))
  cdef np.ndarray result = Xs.sum(0)
  cdef double z = 0.0

  for i in range(n-1,-1,-1):
    buf.fill(0.0)
    z = 0.0
    for j in range(n-1,i-1,-1):
      buf += Xs[j] * cs[j]
      z += cs[j]
    result -= buf / z
  return result

def plackett_luce_pdf(np.ndarray[double, ndim=1] v):
  return exp(plackett_luce_log_pdf(v))
