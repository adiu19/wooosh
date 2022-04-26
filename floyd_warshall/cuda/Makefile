NVCC= nvcc            
execname= fwcu

$(execname): floyd_warshall_cuda.cu
	nvcc -std=c++11 -o $(execname) floyd_warshall_cuda.cu -lm


clean:
	rm $(execname)