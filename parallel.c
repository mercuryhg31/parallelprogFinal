#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

// extern int mandelbrot(Complex c, unsigned int iterations, int blockNumb, int threadsNumb);

typedef struct {
    double real;
    double imag;
} Complex;

// /*
int mandelbrot(Complex c, int max_iter) {
    Complex z = {0, 0};

    for (int i = 0; i < max_iter; ++i) {
        double z_real_sq = z.real * z.real;
        double z_imag_sq = z.imag * z.imag;

        if (z_real_sq + z_imag_sq > 4.0) {
            return i; // escaped
        }

        double z_real_temp = z_real_sq - z_imag_sq + c.real;
        z.imag = 2.0 * z.real * z.imag + c.imag;
        z.real = z_real_temp;
    }

    return max_iter; // didn't escape
}
// */

int* test = NULL; // only for debugging
Complex* grandata = NULL;
char* grandresult = NULL; // only for debugging
FILE* file = NULL;
MPI_File fh;
MPI_Offset dataStart;
MPI_Datatype COMPLEX;

static inline void init(int size, int max_iter) {
    file = fopen("mandelbrot.ppm", "wb");
    fprintf(file, "P6\n%d %d\n255\n", size, size);

    char string[1024];
    snprintf(string, sizeof(string), "P6\n%d %d\n255\n", size, size);
    const int len = strlen(string);
    printf("string len = %d\n", len);
    MPI_File_write(fh, string, len, MPI_CHAR, MPI_STATUS_IGNORE);
    dataStart = len;
    MPI_File_seek(fh, dataStart, MPI_SEEK_SET);

    test = calloc(size * size, sizeof(int));
    grandata = calloc(size * size, sizeof(Complex));
    grandresult = calloc(size * size, sizeof(char));

    MPI_Offset offset = dataStart;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int i = size * y + x;
            test[i] = i;
            grandata[i] = (Complex) {
                -2.0 + (3.0  * x) / (double) size,
                -1.5 + (3.0 * y) / (double) size
            };

            int iterations = mandelbrot(grandata[i], max_iter);
            char colour = (int)(255 * (1.0 - (double)iterations / max_iter));

            // need thrice to write RGB
            fputc(colour, file);
            fputc(colour, file);
            fputc(colour, file);

            char c[3] = {colour, colour, colour};
            // MPI_File_write(fh, c, 3, MPI_CHAR, MPI_STATUS_IGNORE);

            MPI_File_seek()
            MPI_File_write_at()
        }
    }
    fclose(file);
}

static inline void rankWork(Complex* recvbuf, int size, int rankSize, int max_iter) {
    for (int i = 0; i < rankSize; ++i) {
        int iterations = mandelbrot(recvbuf[i], max_iter);
        char colour = (int)(255 * (1.0 - (double)iterations / max_iter));

        char c[3] = {colour, colour, colour};
        MPI_File_write(fh, c, 3, MPI_CHAR, MPI_STATUS_IGNORE);
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
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    const int size = atoi(argv[1]);
    const int max_iter = atoi(argv[2]);

    if (size % numranks != 0) {
        if (myrank == 0) printf("ERROR: Image size not divisible among set number of ranks.");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    MPI_Type_contiguous(2, MPI_DOUBLE, &COMPLEX);
    MPI_Type_commit(&COMPLEX);
    // int count = 2; //number of elements in struct
    // MPI_Aint offsets[count] = {0, 8};
    // int blocklengths[count] = {1, 1};
    // MPI_Datatype types[count] = {MPI_FLOAT, MPI_CHAR};
    // MPI_Datatype my_mpi_type;

    // MPI_Type_create_struct(count, blocklengths, offsets, types, &my_mpi_type);

    MPI_File_open(MPI_COMM_WORLD, "mandelbrot-mpi.ppm", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

    const int rankNumPixels = size * size / numranks;
    Complex* localdata = calloc(rankNumPixels, sizeof(Complex));

    if (myrank == 0) {
        init(size, max_iter);

    }
    // MPI_Scatter(grandata, rankNumPixels, COMPLEX, localdata, rankNumPixels, COMPLEX, 0, MPI_COMM_WORLD);
    // rankWork(localdata, size, rankNumPixels, max_iter);

    printf("Rank %d: Mandelbrot image generated successfully.\n", myrank);
    free(test);
    free(grandata);
    free(grandresult);
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}
