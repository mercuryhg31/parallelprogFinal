#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> // pulls in declaration of malloc, free
#include <string.h> // pulls in declaration for strlen.
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

typedef struct {
    double real;
    double imag;
} Complex;
 

int* test = NULL; // only for debugging
Complex* grandata = NULL;
char* grandresult = NULL; // only for debugging
FILE* file = NULL;

MPI_File fh;

MPI_Offset dataStart;


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

    test = calloc((size * size), sizeof(int));
    grandata = calloc(size * size, sizeof(Complex));
    grandresult = calloc(size * size, sizeof(char));

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
            MPI_Request rq;
            MPI_File_iwrite(fh, c, 3, MPI_CHAR, &rq);

        }
    }
    fclose(file);
}

static inline void generateImage(Complex *subimage, unsigned int max_iter, unsigned int threads, int sizeSubworld, int myrank, int worldSize){
    printf("Launching NOT cuda...\n");
    // int **iterArray;
    // iterArray  = (int **)malloc( sizeof(int *));
    // mandelbrot(subimage, iterations, threads, sizeSubworld, iterArray, myrank, worldSize);

    MPI_Offset rankStart = dataStart + (sizeSubworld * myrank);
    printf("RANKSTART AT %d:: %lld\n", myrank, rankStart);

    MPI_File_seek(fh, rankStart, MPI_SEEK_SET);
    for (int i = 0; i < sizeSubworld; i++) {
        // printf("real: %f, imag: %f\n", subimage[i].real, subimage[i].imag);
        int iterations = mandelbrot(subimage[i], max_iter);
        char colour = (int)(255 * (1.0 - (double)iterations / max_iter));

        char c[3] = {colour, colour, colour};
        MPI_Request rq;
        // MPI_File_write(fh, c, 3, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_iwrite(fh, c, 3, MPI_CHAR, &rq);
    }
    printf("We r out!\n\n");
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

    // if (sqrt(numranks) - (int) sqrt(numranks) != 0.0) {
    //     if (myrank == 0) printf("ERROR: Not running on a square number of ranks.");
    //     MPI_Finalize();
    //     exit(EXIT_FAILURE);
    // }

    // const int root = sqrt(numranks);

    // if (size % root != 0) {
    //     if (myrank == 0) printf("ERROR: Image not divisible among set number of ranks.");
    //     MPI_Finalize();
    //     exit(EXIT_FAILURE);
    // }

    if (size % numranks != 0) {
        if (myrank == 0) printf("ERROR: Image size not divisible among set number of ranks.");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    MPI_File_open(MPI_COMM_WORLD, "mandelbrot1-mpi.ppm", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

    // // int rankSize = 
    // Complex c;
    // MPI_Datatype typesig[2] = {MPI_DOUBLE, MPI_DOUBLE};
    // int block_lengths[2] = {1, 1};
    // MPI_Aint displacements[2];
    // MPI_Get_address(&c.real, &displacements[0]);
    // MPI_Get_address(&c.imag, &displacements[1]);
    // // create and commit the new type
    // MPI_Datatype mpi_complex;
    // MPI_Type_create_struct(2, block_lengths, displacements, typesig, &mpi_complex);
    // MPI_Type_commit(&mpi_complex);
    
    // //create image in 1d array
    
    //create buffer to receive the 1d array
    const int sizeSubworld = (size*size)/numranks;
    Complex *localdata = calloc(sizeSubworld, sizeof(Complex));

    // char string[1024];
    // snprintf(string, sizeof(string), "P6\n%d %d\n255\n", size, size);
    // const int len = strlen(string);
    // printf("string len = %d\n", len);

    // MPI_File_write(fh, string, len, MPI_CHAR, MPI_STATUS_IGNORE);

    if (myrank == 0) {
        init(size, max_iter);
        printf("Scattering...\n");
    }
    else printf("Receiving...\n");

    const int rankBytes = sizeSubworld * sizeof(Complex);
    MPI_Scatter(grandata, rankBytes, MPI_BYTE, localdata, rankBytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    printf("size of subworld: %d rank: %d\n", sizeSubworld, myrank);
    // generateImage(localdata, max_iter, 1024,  sizeSubworld, myrank, size);

    free(localdata);
    free(test);
    free(grandata);
    free(grandresult);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);

    MPI_Finalize();
    // return 0;
}
