NVCC= nvcc            
execname= bscu

$(execname): bitonic_sort_cuda.cu
	nvcc -std=c++11 -o $(execname) bitonic_sort_cuda.cu -lm


clean:
	rm $(execname)



