
Implementation of a Bitonic Parallel Sort Algorithm

Input data(for smaller inputs) is provided and the script to generate the same is also provided in the input_data folder.
Please maintain the same directory structure for scripts to run smoothly.

Input Data generation :
cd input_data/
g++ -std=c++11 generate_data.cpp -o generate_data
./generate_data 2048

Serial :
cd serial/
g++ -std=c++11 bitonic_sort_serial.cpp -o bsseq
./bsseq 2048

CUDA version :
cd cuda/
nvcc -std=c++11 -O1 -o bscu bitonic_sort_cuda.cu -Xcompiler -lrt -lm
./bscu 2048

OpenCL :
cd opencl
g++ -std=c++11 bitonic_sort_opencl.cpp -o bsocl -I/usr/local/cuda/include/ -lOpenCL
./bsocl 2048
