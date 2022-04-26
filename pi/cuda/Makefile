NVCC= nvcc            
execname= picu

$(execname): pi-optimized.cu
	nvcc -o $(execname) -std=c++11 pi-optimized.cu -lm 


clean:
	rm $(execname)
