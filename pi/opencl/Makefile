GPP= g++
execname= piocl

$(execname): pi_opencl.cpp
	g++ -g pi_opencl.cpp -std=c++11 -o $(execname) -I/usr/local/cuda/include/ -I. -lOpenCL 

clean:
	rm $(execname)