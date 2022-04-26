NVCC= nvcc            
execname= kmcu

$(execname): kmeans.cu
	nvcc -o $(execname) -std=c++11 kmeans.cu -lm 


clean:
	rm $(execname)
