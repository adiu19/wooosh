all:
	make -C pi/cuda
	make -C pi/opencl
	make -C pi/seq
	make -C pagerank/cuda
	make -C pagerank/seq
	make -C pagerank/opencl-revamped
clean:
	make clean -C pi/cuda
	make clean -C pi/opencl
	make clean -C pi/seq
	make clean -C pagerank/cuda
	make clean -C pagerank/seq
	make clean -C pagerank/opencl-revamped