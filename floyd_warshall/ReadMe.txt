An output folder should be created inside serial, cuda_version and opencl where outputs will be stored.
Input data is provided and the script to generate is provided in the same folder

Input Data generation :
g++ -std=c++11 matGen.cpp -o matGen
./matGen 2048

serial :
take input data from input_data
g++ -std=c++11 floyd_warshall_serial.cpp -o fwserial
./fwserial 2048

cuda version :
nvcc -std=c++11 -O1 -o fwcuda floyd_warshall_cuda.cu -Xcompiler -lrt -lm
./fwcuda 128

opencl :
g++ -std=c++11 floyd_warshall_opencl.cpp -o fwopencl -I/usr/local/cuda/include/ -lOpenCL
./fwopencl 4096
