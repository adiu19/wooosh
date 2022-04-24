objects= pi_cuda.o kernels.o 
NVCC= nvcc            
execname= picu

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

kernels.o: kernels.cu
	$(NVCC) -c kernels.cu
pi_cuda.o: pi_cuda.cu
	$(NVCC) -std=c++11 -c pi_cuda.cu

clean:
	rm $(objects)
	rm $(execname)


#Actual Commands that are run (linkage error : https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference)
#
#nvcc             -std=c++11 -c pi_cuda.cu
#nvcc             -c kernels.cu
#nvcc              -o picu pi_cuda.o kernels.o
