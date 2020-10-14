#!/usr/bin/env python

import numpy as np
from misc import *
import matplotlib.pyplot as plt

from lalapps import pulsarpputils as pppu

"""
Script to try a toy ROQ model for a line
"""

def signalmodel(t, m, c):
  """
  A line
  """

  return (m*t + c)

# a time series
t0 = 0
tend = 10
N = 20
ts = np.linspace(t0, tend, N)
dt = ts[1]-ts[0]

# number of training waveforms
TS_size = 100
ms = np.random.rand(TS_size)*5.-10.
cs = np.random.rand(TS_size)*5.-10.

# allocate memory and create training set
TS = np.zeros(TS_size*len(ts)).reshape(TS_size, len(ts)) # store training space in TS_size X len(ts) array

A = 1.

for i in range(TS_size):
  TS[i] = signalmodel(ts, ms[i], cs[i])

  # print TS[i]
  
  # normalize
  TS[i] /= np.sqrt(abs(dot_product(dt, TS[i], TS[i])))
  
# Allocate storage for projection coefficients of training space waveforms onto the reduced basis elements
proj_coefficients = np.zeros(TS_size*TS_size).reshape(TS_size, TS_size)

# Allocate matrix to store the projection of training space waveforms onto the reduced basis 
projections = np.zeros(TS_size*len(ts)).reshape(TS_size, len(ts)) 

rb_errors = []

#### Begin greedy: see Field et al. arXiv:1308.3565v2 #### 

tolerance = 10e-12 # set maximum RB projection error

sigma = 1 # (2) of Algorithm 1. (projection error at 0th iteration)

rb_errors.append(sigma)

RB_matrix = [TS[0]] # (3) of Algorithm 1. (seed greedy algorithm (arbitrary))

iter = 0

#print TS

while sigma >= tolerance: # (5) of Algorithm 1.
  # project the whole training set onto the reduced basis set
  projections = project_onto_basis(dt, RB_matrix, TS, projections, proj_coefficients, iter) 

  residual = TS - projections
  # Find projection errors
  projection_errors = [dot_product(dt, residual[i], residual[i]) for i in range(len(residual))]
  
  print projection_errors
  
  sigma = abs(max(projection_errors)) # (7) of Algorithm 1. (Find largest projection error)
  print sigma, iter
  index = np.argmax(projection_errors) # Find Training-space index of waveform with largest proj. error 

  rb_errors.append(sigma)
    
  #Gram-Schmidt to get the next basis and normalize

  print index
  
  next_basis = TS[index] - projections[index] # (9) of Algorithm 1. (Gram-Schmidt)
  next_basis /= np.sqrt(abs(dot_product(dt, next_basis, next_basis))) #(10) of Alg 1. (normalize)

  RB_matrix.append(next_basis) # (11) of Algorithm 1. (append reduced basis set)

  iter += 1
  
#print TS
  
#### Error check ####
"""
TS_rand_size = 100

TS_rand = np.zeros(TS_rand_size*len(ts)).reshape(TS_rand_size, len(ts)) # Allocate random training space

psis_rand = np.random.rand(TS_rand_size)*(np.pi/2.)-(np.pi/4.)
phi0s_rand = np.random.rand(TS_rand_size)*(2.*np.pi)

for i in range(TS_rand_size):
  TS_rand[i] = signalmodel(ts, A, phi0s_rand[i], psis_rand[i], ra, dec, det)
  # normalize
  TS_rand[i] /= np.sqrt(abs(dot_product(dt, TS_rand[i], TS_rand[i])))


### find projection errors ###
iter = 0
proj_rand = np.zeros(len(ts))
proj_error = []

for h in TS_rand:
  while iter < len(RB_matrix):
    proj_coefficients_rand = dot_product(dt, RB_matrix[iter], h)
    proj_rand += proj_coefficients_rand*RB_matrix[iter]

    iter += 1

  residual = h - proj_rand
  projection_errors = abs(dot_product(dt, residual, residual))
  proj_error.append(projection_errors)
  proj_rand = np.zeros(len(ts))
        
  iter = 0
        
plt.scatter(np.linspace(0, len(proj_error), len(proj_error)), np.log10(proj_error))
plt.ylabel('log10 projection error')
plt.show()
"""