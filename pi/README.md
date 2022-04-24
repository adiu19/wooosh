g++ -std=c++11 -o pi_seq pi_seq.cpp -I.
nvcc -o pi_cuda pi_cuda.cu -I/usr/local/cuda/include/ -lm
