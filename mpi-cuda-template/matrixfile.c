#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdint.h>

#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

uint64_t clock_now(void)
{
   unsigned int tbl, tbu0, tbu1;

   do {
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
      __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
   } while (tbu0 != tbu1);
   return (((uint64_t)tbu0) << 32) | tbl;
}
#endif

void multiplyMatrix(float **mat1, float **mat2, float **res, int N)
{
   int i, j, k;
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         res[i][j] = 0.0f;
         for (k = 0; k < N; k++)
            res[i][j] += mat1[i][k] * mat2[k][j];
      }
   }
}

void displayMatrix(float *mat, int r, int c)
{
   int i, j;
   for (i = 0; i < r; i++) {
      for (j = 0; j < c; j++)
         printf("%.0f ", mat[j + i * c]);
      printf("\n");
   }
}

void printArray (float *row, int nElements) {
    int i;
    for (i=0; i<nElements; i++) {
        printf("%.0f ", row[i]);
    }
    printf("\n");
}

//#define OUTPUT_MATRIX
#define IO
#define FILENAME "matrix_multiply_result.txt"

int main(int argc, char **argv)
{
   // Initialize the MPI environment
   MPI_Init(&argc, &argv);

   // Get the number of processes
   int numProcs;
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

   // Get the rank of the process
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   double start_time, end_time;
   if (argc != 2) {
      printf("HighLife requires 1 argument\n");
      exit(-1);
   }
   int matrixSize = atoi(argv[1]);
   if (matrixSize % numProcs != 0) {
      printf("Matrix size must be divisible by the number of processes\n");
      exit(-1);
   }
   int rowsPerProc = matrixSize / numProcs;
   size_t cellsPerProc = rowsPerProc * matrixSize;
   int i, j, k;
   float *a, *b, *c;
   uint64_t *clocks;
   b = calloc(matrixSize * matrixSize, sizeof(float));
   float *sentRow = calloc(cellsPerProc, sizeof(float));
   uint64_t clock_before_mult, clock_after_mult;
   if (myRank == 0) {
      a = calloc(matrixSize * matrixSize, sizeof(float));

      c = calloc(matrixSize * matrixSize, sizeof(float));
      clocks = calloc(numProcs, sizeof(uint64_t));

      float n = 0.0f;
      for (i = 0; i < matrixSize*matrixSize; i++) {
         a[i] = n++;
         b[i] = n++;
      }
      #ifdef OUTPUT_MATRIX
         displayMatrix(a, matrixSize, matrixSize);
      #endif
      start_time = MPI_Wtime();

   }
   // scatter matrix A based on size and number of processors
   MPI_Scatter(a, cellsPerProc, MPI_FLOAT, sentRow, cellsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
   // send b matrix to all processes
   MPI_Bcast(b, matrixSize * matrixSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

   // printf("Process %d received elements: ", myRank);
   // printArray(sentRow, cellsPerProc);
   // displayMatrix(b, matrixSize);
   MPI_Barrier(MPI_COMM_WORLD);
   // compute vector multiplication of sentRow with b
   float *result = calloc(cellsPerProc, sizeof(float));
   if (myRank == 0) clock_before_mult = clock_now();
   for (i = 0; i < rowsPerProc; i++) { // loop through the rows each processor is assigned
      for (j = 0; j < matrixSize; j++) { // column offset of c and b
         for (k = 0; k < matrixSize; k++) // column offset of a, row offset of b
            result[j + (i * matrixSize)] += sentRow[i * matrixSize + k] * b[k * matrixSize + j];
      }
   }
   if (myRank == ) clock_after_mult = clock_now();

   // printf("Process %d computed elements: ", myRank);
   MPI_Barrier(MPI_COMM_WORLD);
   // gather results from all processes
   MPI_Gather(result, cellsPerProc, MPI_FLOAT, c, cellsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
   end_time = MPI_Wtime();
   if (myRank == 0) {
      uint64_t clock_diff = clock_after_mult - clock_before_mult;

      printf("Number of cycles for processes to multiply matricies: %lu", clock_diff);
      printf("Parallel multiplication\n");
      #ifdef OUTPUT_MATRIX
         displayMatrix(c, matrixSize, matrixSize);
      #endif
      free(a);
      free(c);
      printf("Time taken for matrix multiplication with %d ranks and %d matrix size: %f seconds\n", numProcs, matrixSize, end_time - start_time);
   }

   free(sentRow);
   free(b);

   remove(FILENAME);

   #ifdef IO
      MPI_Barrier(MPI_COMM_WORLD);

      // MPI I/O
      double start_time_IO, end_time_IO;
      uint64_t before_write_at, after_write_at;
      // MPI I/O to write chunk to file
      MPI_File fh;
      MPI_Status status;

      if (myRank == 0) {
         printf("\nMPI I/O\n");
      }

      MPI_File_open(MPI_COMM_WORLD, FILENAME,
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

      if (myRank == 0) {
         start_time_IO = MPI_Wtime();
         before_write_at = clock_now();
      }

      MPI_File_set_atomicity(fh, 1);

      MPI_File_write_at(fh, myRank * cellsPerProc * sizeof(float), result, cellsPerProc, MPI_FLOAT, &status);
      if(myRank == 0) after_write_at = clock_now();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myRank == 0) {
         // MPI I/O to read results from file
         float* c_fromFile = calloc(matrixSize * matrixSize, sizeof(float));
         // MPI_File_read_at(fh, 0, c_fromFile, matrixSize * matrixSize, MPI_FLOAT, &status);
         MPI_File_close(&fh);

         end_time_IO = MPI_Wtime();
         printf("Processes wrote chunks to file and process %d read from file.\n", myRank);
         #ifdef OUTPUT_MATRIX
            displayMatrix(c_fromFile, matrixSize, matrixSize);
         #endif
         free(c_fromFile);
         printf("Time taken for %d processes to write chunks to file and process %d to read from file: %f seconds\n",
               numProcs, myRank, end_time_IO - start_time_IO);
      } else {
         MPI_File_close(&fh);
      }




      if (myRank == 0) {
         uint64_t write_at_cycles = after_write_at - before_write_at;
         printf("Number of cycles for processes to write chunks to file: %lu", sum_write_at_cycles);
      }

   #endif

   free(result);
   MPI_Finalize();
   return 0;
}