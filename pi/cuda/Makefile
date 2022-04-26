NVCC= nvcc            
execname= pr

$(execname): pi-optimized.cu
	nvcc -o pr -std=c++11 pi-optimized.cu -lm 


clean:
	rm $(execname)
