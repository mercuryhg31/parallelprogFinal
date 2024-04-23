#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

typedef struct {
    double real;
    double imag;
} Complex;

extern "C" 
{
    void mandelbrot(Complex *c, unsigned int iterations, int threadsNumb, int sizeSubworld, int **res);
}

__global__ void mandelbrotKernel(Complex *c, unsigned int iterations, int *results);


void mandelbrot(Complex *c, unsigned int iterations, int threadsNumb, int sizeSubworld, int **res){ 
   
    Complex *devComplex;
    cudaMalloc(&devComplex, sizeSubworld * sizeof(Complex));

    // Copy data from host to device
    cudaMemcpy(devComplex, c, sizeSubworld * sizeof(Complex), cudaMemcpyHostToDevice);

    
    int *results;
    cudaMallocManaged( &results, sizeSubworld * sizeof(int));
    cudaMemset (results, 0,  sizeSubworld * sizeof(int));
    mandelbrotKernel<<<sizeSubworld/threadsNumb,threadsNumb>>>(devComplex, iterations, results); //blocks = arraySize/threads and the threadcount is specified by user
    
    
    //copy results back to host
    int *hostResults;
    hostResults = (int*) malloc(sizeSubworld * sizeof(int));
    printf("We are inside CUDA!\n\n");
    // Copy data from device to host
    cudaMemcpy(results, hostResults, sizeSubworld * sizeof(int), cudaMemcpyDeviceToHost);
    printf("after copy!\n\n");
    *res = hostResults;
    printf("after results!\n\n");
    cudaDeviceSynchronize(); //sync final result 
    cudaFree(devComplex);
    cudaFree(results);
    

}


__global__ void mandelbrotKernel(Complex *c, unsigned int iterations, int *results) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    Complex z = {0, 0};
    int res = iterations;
    for (int i = 0; i < iterations; i++) {
        double z_real_sq = z.real * z.real;
        double z_imag_sq = z.imag * z.imag;

        if (z_real_sq + z_imag_sq > 4.0) {
            res = i; // escaped
            break;
        }

        double z_real_temp = z_real_sq - z_imag_sq + (*c).real;
        z.imag = 2.0 * z.real * z.imag + (*c).imag;
        z.real = z_real_temp;
    }

    results[idx] = res; // didn't escape (Max_iterartions - 1)
}

    
