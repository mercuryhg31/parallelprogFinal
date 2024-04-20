#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
    int mandelbrot(Complex c, unsigned int iterations, int blockNumb, int threadsNumb);
}


__global__ typedef struct {
    double real;
    double imag;
} Complex;


int mandelbrot(Complex **c, unsigned int iterations, int threadsNumb){
    size_t array_size = sizeof(c);
    Complex *deviceArray = NULL; //copy host array (complex *c) to device 

    //allocate memory on GPU for the host data
    cudaMalloc((void **)&deviceArray, array_size * sizeof(Complex));
    cudaMemcpy(deviceArray, *c, array_size * sizeof(Complex), cudaMemcpyHostToDevice);

    mandelbrotKernel<<<array_size/threadsNumb,threadsNumb>>>(deviceArray, iterations); //blocks = arraySize/threads and the threadcount is specified by user
    cudaDeviceSynchronize(); //sync final result 
    cudaMemcpy(*c, deviceArray, N * sizeof(Complex), cudaMemcpyDeviceToHost); //copy device array back to host

    cudaFree(deviceArray);
}


__global__ int mandelbrotKernel(Complex *c, unsigned int iterations) {
    Complex z = {0, 0};

    for (int i = 0; i < iterations; i++) {
        double z_real_sq = z.real * z.real;
        double z_imag_sq = z.imag * z.imag;

        if (z_real_sq + z_imag_sq > 4.0) {
            return i; // escaped
        }

        double z_real_temp = z_real_sq - z_imag_sq + c.real;
        z.imag = 2.0 * z.real * z.imag + c.imag;
        z.real = z_real_temp;
    }

    return iterations; // didn't escape (Max_iterartions - 1)
}

    
