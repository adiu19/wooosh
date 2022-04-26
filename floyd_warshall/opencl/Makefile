GPP= g++
execname= fwocl

$(execname): floyd_warshall_opencl.cpp
	g++ -std=c++11 floyd_warshall_opencl.cpp -o $(execname) -I/usr/local/cuda/include/ -lOpenCL
clean:
	rm $(execname)
