MODULES := pi/cuda pi/opencl pi/seq pagerank/cuda pagerank/seq pagerank/opencl floyd_warshall/serial floyd_warshall/cuda floyd_warshall/opencl bitonic_sort/serial bitonic_sort/cuda bitonic_sort/opencl kmeans/cuda kmeans/opencl kmeans/serial
all:
	$(foreach mod,$(MODULES),make -C $(mod);)

clean:
	$(foreach mod,$(MODULES),make clean -C $(mod);)