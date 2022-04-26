Implementation of a Simplified KMeans Algorithm.

Serial:
Build : g++ -std=c++11 kmeansSeq.cpp -o kmseq -lm
Run : ./kmseq


OpenCL:
Build : g++ -std=c++11 kmeans.cpp -o kmocl -I/usr/local/cuda/include/ -l OpenCL
Run : ./kmocl


CUDA:
Build : nvcc kmeans.cu -o kmcu
Run : ./kmcu