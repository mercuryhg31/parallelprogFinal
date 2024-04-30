#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

typedef struct {
    double real;
    double imag;
} Complex;

extern "C" 
{
    void mandelbrot(Complex *c, unsigned int iterations, int threadsNumb, int sizeSubworld, int **res, int myrank, int worldSize);
}

__global__ void mandelbrotKernel(Complex *c, unsigned int iterations, int *results, int myrank, int sizeSubworld, int worldSize);


void mandelbrot(Complex *c, unsigned int iterations, int threadsNumb, int sizeSubworld, int **res, int myrank, int worldSize){ 
   
    Complex *devComplex;
    cudaMalloc(&devComplex, sizeSubworld * sizeof(Complex));

    // Copy data from host to device
    cudaMemcpy(devComplex, c, sizeSubworld * sizeof(Complex), cudaMemcpyHostToDevice);

    
    int *results;
    cudaMallocManaged( &results, sizeSubworld * sizeof(int));
    cudaMemset (results, 0,  sizeSubworld * sizeof(int));
    printf("subworldSize: %d\n", sizeSubworld);
    printf("Space: %d   Blocks: %f\n", sizeof(c), ceil(sizeSubworld/threadsNumb)+1);

    mandelbrotKernel<<<(sizeSubworld + threadsNumb - 1) / threadsNumb, threadsNumb>>>(devComplex, iterations, results, myrank, sizeSubworld, worldSize); //blocks = arraySize/threads and the threadcount is specified by user
    
    
    //copy results back to host
    int *hostResults;
    hostResults = (int*) malloc(sizeSubworld * sizeof(int));
    printf("We are inside CUDA!\n\n");
    // Copy data from device to host
    cudaMemcpy(hostResults, results, sizeSubworld * sizeof(int), cudaMemcpyDeviceToHost);
    printf("after copy!\n\n");
    // for (int i = 0; i < sizeSubworld; i++){
    //     printf("%d\n", hostResults[i]);
    // }
    *res = hostResults;
    printf("after results!\n\n");
    cudaDeviceSynchronize(); //sync final result 
    cudaFree(devComplex);
    cudaFree(results);


}


__global__ void mandelbrotKernel(Complex *c, unsigned int iterations, int *results, int myrank, int sizeSubworld, int worldSize) {

    int idx = (threadIdx.x+blockDim.x*blockIdx.x);
    if( idx < sizeSubworld){
        
        idx = idx + (myrank*sizeSubworld);
        int x;
        int y;
   
        x = idx % (worldSize); // Calculate x coordinate
        y = idx / (worldSize); // Calculate y coordinate
        //printf(" myrank: %d idxb: %d idx: %d  worldSize: %d     x: %d   y: %d\n", myrank, idx-(myrank*sizeSubworld),idx, worldSize , x, y);
   
        Complex z = {0, 0};
        Complex f = {
                    -2.0 + (3.0  * x) / (double) worldSize,
                    -1.5 + (3.0 * y) / (double) worldSize
                };
        unsigned int res = iterations;

        for (int i = 0; i < iterations; i++) {
            double z_real_sq = z.real * z.real;
            double z_imag_sq = z.imag * z.imag;

            if (z_real_sq + z_imag_sq > 4.0) {
                res = i; // escaped
                break;
            }

            double z_real_temp = z_real_sq - z_imag_sq + f.real;
            z.imag = 2.0 * z.real * z.imag + f.imag;
            z.real = z_real_temp;
        }
        
        // printf("iterations: %d\n", res);
        

        results[idx-(myrank*sizeSubworld)] = res; // didn't escape (Max_iterartions - 1)
        // char colour = (int)(255 * (1.0 - (double)iterArray[0][i] / iterations));
        // char c[3] = {colour, colour, colour};
        // MPI_File_write(fh, c, 3, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    
}

    
