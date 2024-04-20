#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
    int mandelbrot(Complex c, int iterations);
    }



__global__ typedef struct {
    double real;
    double imag;
} Complex;


int mandelbrot(Complex c, unsigned int iterations){
    for 
}




__global__ int mandelbrotKernel(Complex c, unsigned int iterations) {
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

    return iterations; // didn't escape
}

    
