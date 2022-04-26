all:
	make -C pi/cuda
	make -C pi/opencl
	make -C pi/seq
	make -C pagerank/cuda
	make -C pagerank/seq
	make -C pagerank/opencl
	make -C floyd_warshall/serial
	make -C floyd_warshall/cuda
	make -C floyd_warshall/opencl
	make -C bitonic_sort/serial
	make -C bitonic_sort/cuda
	make -C bitonic_sort/opencl
	make -C kmeans/cuda
	make -C kmeans/opencl
	make -C kmeans/serial
clean:
	make clean -C pi/cuda
	make clean -C pi/opencl
	make clean -C pi/seq
	make clean -C pagerank/cuda
	make clean -C pagerank/seq
	make clean -C pagerank/opencl
	make clean -C floyd_warshall/serial
	make clean -C floyd_warshall/cuda
	make clean -C floyd_warshall/opencl
	make clean -C bitonic_sort/serial
	make clean -C bitonic_sort/cuda
	make clean -C bitonic_sort/opencl
	make clean -C kmeans/cuda
	make clean -C kmeans/opencl
	make clean -C kmeans/serial