GPP= g++
execname= kmocl

$(execname): kmeans.cpp
	g++ -g kmeans.cpp -std=c++11 -o $(execname) -I/usr/local/cuda/include/ -I. -lOpenCL 

clean:
	rm $(execname)