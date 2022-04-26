Implementation of Page Rank Algorithm : Sequential, CUDA, and OpenCL.

The "data" folder contains a few data files to help run the code. The code, as an output, prints the top 10 ranked nodes in the graph.


Build Commands (to be run in the individual folders):
    CUDA : nvcc -o prcu pagerank.cu -lm
    Sequential : g++ -std=c++11 -o prseq pagerank.cpp -I.
    OpenCL : g++ -std=c++11 -o procl pagerank.cpp -I/usr/local/cuda/include/ -I. -lOpenCL

Run Commands  (to be run in the individual folders):
    CUDA : ./prcu ../data/100.txt
    Seq : ./prseq ../data/100.txt
    OpenCL : ./procl ../data/100.txt

