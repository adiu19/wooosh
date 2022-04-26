serial:
gcc -std=c++11 kmeansSeq.cpp -o kmeansSeq -lm
./kmeansSeq


opencl:
gcc -std=c++11 kmeans.cpp -o kmeanscl -I/usr/local/cuda/include/ -l OpenCL
./kmeanscl


cuda:
nvcc kmeans.cu -o kmeansCU
./kmeansCU