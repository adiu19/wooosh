execname= procl

$(execname): pagerank.cpp
	g++ -std=c++11 -o procl pagerank.cpp -I/usr/local/cuda/include/ -I. -lOpenCL

clean:
	rm $(execname)