GPP= g++
execname= bsocl

$(execname): bitonic_sort_opencl.cpp
	g++ -std=c++11 bitonic_sort_opencl.cpp -o $(execname) -I/usr/local/cuda/include/ -lOpenCL
clean:
	rm $(execname)
