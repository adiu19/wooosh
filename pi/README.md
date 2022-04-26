Implementation of Monte Carlo Simulation to estimate PI : Sequential, CUDA, and OpenCL. The number of simulations are embedded in the code.

Build Commands (to be run in the individual folders. not a necessary step since we have provided a make file for each):
    CUDA : nvcc -o picu -std=c++11  pi-optimized.cu -lm
    Sequential : g++ -std=c++11 -o piseq pi_seq.cpp -I.
    OpenCL : g++ -std=c++11 -o piocl pi_opencl.cpp -I/usr/local/cuda/include/ -I. -lOpenCL

Run Commands  (to be run in the individual folders):
    CUDA : ./picu
    Seq : ./piseq
    OpenCL : ./piocl

