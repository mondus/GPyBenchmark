echo on
call activate GPy
SET CUDA_DEVICE=2
SET PYCUDA_DEFAULT_NVCC_FLAGS=-lineinfo
python rbf_benchmark.py
deactivate