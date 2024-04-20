#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


extern "C" 
{
void runCudaLand( int myrank );
void sort(int *arr, int n, bool direction);
}


__global__ void Hello_kernel( int myrank );

void runCudaLand( int myrank )
{
  printf("MPI rank %d: leaving CPU land \n", myrank );

  cudaSetDevice( myrank % 4 );

  Hello_kernel<<<128,128>>>( myrank );

  printf("MPI rank %d: re-entering CPU land \n", myrank );
}

void sort(int *arr, int n, bool direction) {
    int *d_arr;
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize(1, 1);
    dim3 blockSize(n, 1);
    for (int j = 2; j <= n; j <<= 1) {
        for (int k = j >> 1; k > 0; k >>= 1) {
            bitonicSortKernel<<<gridSize, blockSize>>>(d_arr, n, k, direction);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

__global__ void bitonicSortKernel(int *arr, int n, int j, bool direction) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = tid ^ j;
    if (k > tid) {
        if ((tid & j) == 0) {
            if ((arr[tid] > arr[k]) == direction) {
                // Swap elements if direction is true
                int temp = arr[tid];
                arr[tid] = arr[k];
                arr[k] = temp;
            }
        }
        else {
            if ((arr[tid] < arr[k]) == direction) {
                // Swap elements if direction is true
                int temp = arr[tid];
                arr[tid] = arr[k];
                arr[k] = temp;
            }
        }
    }
}




__global__ void Hello_kernel( int myrank )
{

  int device;

  cudaGetDevice( &device );

  printf("Hello World from CUDA/MPI: Rank %d, Device %d, Thread %d, Block %d \n",
	 myrank, device, threadIdx.x, blockIdx.x );
}
