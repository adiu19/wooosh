Implementation of a Simplified KMeans Algorithm.

The program is defined for K=8 (centroids) and N = 640000 (datapoints). The Input data(for N= 640000 inputs) is provided data folder. 

Serial:
Build : g++ -std=c++11 kmeansSeq.cpp -o kmseq -lm
Run : ./kmseq


OpenCL:
Build : g++ -std=c++11 kmeans.cpp -o kmocl -I/usr/local/cuda/include/ -l OpenCL
Run : ./kmocl


CUDA:
Build : nvcc kmeans.cu -o kmcu
Run : ./kmcu