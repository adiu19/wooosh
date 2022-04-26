execname= prcu

$(execname): pagerank.cu
	nvcc -o pr pagerank.cu -lm 

clean:
	rm $(execname)