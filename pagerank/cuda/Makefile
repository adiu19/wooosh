execname= prcu

$(execname): pagerank.cu
	nvcc -o prcu pagerank.cu -lm 

clean:
	rm $(execname)