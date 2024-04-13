#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITERATIONS 1000

typedef struct {
    double real;
    double imag;
} Complex;

int mandelbrot(Complex c) {
    Complex z = {0, 0};

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double z_real_sq = z.real * z.real;
        double z_imag_sq = z.imag * z.imag;

        if (z_real_sq + z_imag_sq > 4.0) {
            return i; // Escaped
        }

        double z_real_temp = z_real_sq - z_imag_sq + c.real;
        z.imag = 2.0 * z.real * z.imag + c.imag;
        z.real = z_real_temp;
    }

    return MAX_ITERATIONS; // Didn't escape
}

int main() {
    FILE* file = fopen("mandelbrot.ppm", "wb");
    fprintf(file, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Complex c = {
                -2.0 + (3.0  * x) / (double) WIDTH,
                -1.5 + (3.0 * y) / (double) HEIGHT
            };

            int iterations = mandelbrot(c);
            int color = (int)(255 * (1.0 - (double)iterations / MAX_ITERATIONS));

            // Need thrice to write RGB
            fputc(color, file);
            fputc(color, file);
            fputc(color, file);
        }
    }

    fclose(file);
    printf("Mandelbrot image generated successfully.\n");

    return 0;
}