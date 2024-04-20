#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

extern int mandelbrot(Complex c, int iterations);

typedef struct {
    double real;
    double imag;
} Complex;

/*
int mandelbrot(Complex c, int iterations) {
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
*/

int* test = NULL;
Complex* grandata = NULL;
Complex* grandresult = NULL;

static inline void init(int size) {
    test = calloc(size * size, sizeof(int));
    grandata = calloc(size * size, sizeof(Complex));
    grandresult = calloc(size * size, sizeof(Complex));

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int i = size * y + x;
            test[i] = i;
            grandata[i] = {
                -2.0 + (3.0  * x) / (double) size,
                -1.5 + (3.0 * y) / (double) size
            };

            // int iterations = mandelbrot(number);
            // int color = (int)(255 * (1.0 - (double)iterations / max_iter));

            // // need thrice to write RGB
            // fputc(color, file);
            // fputc(color, file);
            // fputc(color, file);
        }
    }
}

int main(int argc, char* argv[]) {
    int _myrank, _numranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &_numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);
    const int numranks = _numranks;
    const int myrank = _myrank;

    if (argc < 3) {
        if (myrank == 0) printf("ERROR: Doesn't have 2 arguments: 1) size of image 2) number of max iterations");
        exit(EXIT_FAILURE);
    }

    const int size = atoi(argv[1]);
    const int max_iter = atoi(argv[2]);

    if (sqrt(numranks) - (int) sqrt(numranks) != 0.0) {
        if (myrank == 0) printf("ERROR: Not running on a square number of ranks.");
        exit(EXIT_FAILURE);
    }

    const int root = sqrt(numranks);

    if (size % root != 0) {
        if (myrank == 0) printf("ERROR: Image not divisible among set number of ranks.");
        exit(EXIT_FAILURE);
    }

    if (myrank == 0) {
        // FILE* file = fopen("mandelbrot.ppm", "wb");
        // fprintf(file, "P6\n%d %d\n255\n", size, size);
        

        init(size);

        fclose(file);
        printf("Mandelbrot image generated successfully.\n");
    }

    return 0;
}