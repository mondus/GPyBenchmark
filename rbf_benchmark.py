#!/usr/bin/env python

import importlib as importlib
import numpy as np
from GPy.core.parameterization.variational import NormalPosterior
from GPy.kern import RBF
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU, PSICOMP_RBF_GPUDEV
import time
import sys
import traceback
import pycuda.driver as cuda


# np.random.seed(123)

N,M,Q = 5000,200,20

X = np.random.randn(N,Q)
X_var = np.random.rand(N,Q)+0.01
Z = np.random.randn(M,Q)
qX = NormalPosterior(X, X_var)

w1 = np.random.randn(N)
w2 = np.random.randn(N,M)
w3 = np.random.randn(M,M)
w3n = np.random.randn(N,M,M)

kern = RBF(Q,ARD=True)

#print("""
#======================================
#RBF psi-statistics benchmark (Python)
#======================================
#""")
#print('N = '+str(N))
#print('M = '+str(M))
#print('Q = '+str(Q))
#print('')

#st_time = time.time()
#r1 = kern.psicomp.psicomputations(kern, Z, qX)
#print('RBF psi-stat computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

#st_time = time.time()
#r2 = kern.psicomp.psiDerivativecomputations(kern, w1, w2, w3, Z, qX)
#print('RBF psi-stat derivative computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

#st_time = time.time()
#r3 = kern.psicomp.psicomputations(kern, Z, qX, return_psi2_n=True)
#print('RBF psi-stat (psi2n) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

#st_time = time.time()
#r4 = kern.psicomp.psiDerivativecomputations(kern, w1, w2, w3n, Z, qX)
#print('RBF psi-stat derivative (psi2n) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

print("""


======================================
RBF psi-statistics benchmark (GPU)
======================================
""")
print('N = '+str(N))
print('M = '+str(M))
print('Q = '+str(Q))
print('')

#cuda.start_profiler()




psicomp_gpu = PSICOMP_RBF_GPU() #creates a new CUDA context

start = cuda.Event()
end = cuda.Event()

start.record()
r1g = psicomp_gpu.psicomputations(kern, Z, qX, return_psi2_n=False)
end.record()
end.synchronize()
print('RBF psi-stat computation time: '+'%.2f'%(start.time_till(end))+' msec.')



psicomp_gpu = PSICOMP_RBF_GPUDEV() #creates a new CUDA context

start = cuda.Event()
end = cuda.Event()

start.record()
r1gdev = psicomp_gpu.psicomputations(kern, Z, qX, return_psi2_n=False)
end.record()
end.synchronize()
print('RBF psi-stat computation time: '+'%.2f'%(start.time_till(end))+' msec.')


#st_time = time.time()
#r2g = psicomp_gpu.psiDerivativecomputations(kern, w1, w2, w3, Z, qX)
#print('RBF psi-stat derivative computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

#st_time = time.time()
#r3g = psicomp_gpu.psicomputations(kern, Z, qX, return_psi2_n=True)
#print('RBF psi-stat (psi2n) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

#st_time = time.time()
#r4g = psicomp_gpu.psiDerivativecomputations(kern, w1, w2, w3n, Z, qX)
#print('RBF psi-stat derivative (psi2n) computation time: '+'%.2f'%(time.time()-st_time)+' sec.')

assert np.all([np.allclose(a,b) for a,b in zip(r1g,r1gdev)])
#assert np.all([np.allclose(a,b) for a,b in zip(r1,r1g)])
#assert np.all([np.allclose(a,b) for a,b in zip(r2,r2g)])
#assert np.all([np.allclose(a,b) for a,b in zip(r3,r3g)])
#assert np.all([np.allclose(a,b) for a,b in zip(r4,r4g)])

cuda.stop_profiler()
