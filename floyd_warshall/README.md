Implementation of Floyd Warshall algorithm for APSP.

Please maintain the same directory structure for scripts to run smoothly.

An output folder should be created inside serial, cuda_version and opencl where outputs will be stored.
Input data(for smaller inputs) is provided and the script to generate the same is also provided in the input_data folder.

Input Data generation :
cd input_data/
g++ -std=c++11 matGen.cpp -o matGen
./matGen 1024

Serial :
cd serial/
mkdir output
g++ -std=c++11 floyd_warshall_serial.cpp -o fwserial
./fwserial 1024

CUDA version :
cd cuda/
mkdir output
nvcc -std=c++11 -O1 -o fwcuda floyd_warshall_cuda.cu -Xcompiler -lrt -lm
./fwcuda 1024

OpenCl :
cd opencl
mkdir output
g++ -std=c++11 floyd_warshall_opencl.cpp -o fwopencl -I/usr/local/cuda/include/ -lOpenCL
./fwopencl 1024
