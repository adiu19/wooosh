nvcc -o pr pagerank.cu -lm
g++ -std=c++11 -o pr pagerank.cpp -I.
g++ -std=c++11 -o pr_ocl pagerank.cpp -I/usr/local/cuda/include/ -I. -lOpenCL