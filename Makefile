mpi:
	mpixlc -O3 parallel.c -o parallel

mpi-cuda: parallel.c parallel.cu
	mpixlc -O3 parallel.c -c -o parallel-xlc.o
	nvcc -O3 -arch=sm_70 parallel.cu -c -o parallel-nvcc.o 
	mpixlc -O3 parallel-xlc.o parallel-nvcc.o -o parallel-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ 
