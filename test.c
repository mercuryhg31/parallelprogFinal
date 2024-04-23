#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    double real;
    double imag;
} Complex;

Complex* grandata = NULL;
Complex* grandresult = NULL; // only for debugging
MPI_Datatype COMPLEX;

static inline void printcomp(Complex n) {
	printf("%f/%f ", n.real, n.imag);
}

static inline void init() {
    grandata = calloc(10, sizeof(Complex));
    grandresult = calloc(10, sizeof(Complex));

	for (int i = 0; i < 10; ++i) {
		grandata[i] = (Complex) {(double) i, (double) i};
	}
}

int main(int argc, char* argv[]) {
    int _myrank, _numranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &_numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);
    const int numranks = _numranks;
    const int myrank = _myrank;

	// https://stackoverflow.com/questions/33618937/trouble-understanding-mpi-type-create-struct
	int count = 2; //number of elements in struct
	int blocklengths[count];
	blocklengths[0] = 1;
	blocklengths[1] = 1;

	MPI_Aint disp[count];
	disp[0] = offsetof(Complex, real);
	disp[1] = offsetof(Complex, imag);
	MPI_Datatype types[count];
	types[0] = MPI_DOUBLE;
	types[1] = MPI_DOUBLE;
    MPI_Type_create_struct(count, blocklengths, disp, types, &COMPLEX);

    int rankNumPixels = 10 / numranks;
	Complex* localdata = calloc(rankNumPixels, sizeof(Complex));

    if (myrank == 0) {
        init();
		for (int i = 0; i < 10; ++i) {
			printf("start: ");
			printcomp(grandata[i]);
			printf("\n");
		}
    }
	MPI_Scatter(grandata, rankNumPixels, COMPLEX, localdata, rankNumPixels, COMPLEX, 0, MPI_COMM_WORLD);
	for (int i = 0; i < rankNumPixels; ++i) {
		grandresult[i] = (Complex) {
			localdata[i].real + 1,
			localdata[i].imag + 1
		};
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(localdata, rankNumPixels, COMPLEX, grandresult, rankNumPixels, COMPLEX, 0, MPI_COMM_WORLD);
	if (myrank == 0) {
		for (int i = 0; i < 10; ++i) {
			printf("end: ");
			printcomp(grandresult[i]);
			printf("\n");
		}
	}

	printf("Mandelbrot image generated successfully.\n");
    free(grandata);
    free(grandresult);
    MPI_Finalize();
    return 0;
}
