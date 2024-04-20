#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
extern void runCudaLand(int myrank);
extern void sort(int *arr, int n, bool direction); 

// Function to merge sorted segments into a single sorted array
void mergeSortedSegments(int *sortedArr, int *mergedArr, int *tempArr, int n, int segmentSize) {
    // Copy the first sorted segment directly to the merged array
    for (int i = 0; i < segmentSize; i++) {
        mergedArr[i] = sortedArr[i];
    }
    // Merge remaining sorted segments into the merged array
    for (int i = 1; i < n; i++) {
        merge(mergedArr, &sortedArr[i * segmentSize], tempArr, i * segmentSize, segmentSize);
        memcpy(mergedArr, tempArr, (i + 1) * segmentSize * sizeof(int));
    }
}

// Function to merge two sorted arrays
void merge(int *a, int *b, int *c, int m, int n) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (a[i] <= b[j]) {
            c[k] = a[i];
            i++;
        } else {
            c[k] = b[j];
            j++;
        }
        k++;
    }
    while (i < m) {
        c[k] = a[i];
        i++;
        k++;
    }
    while (j < n) {
        c[k] = b[j];
        j++;
        k++;
    }
}

int main(int argc, char **argv) // size of image dimension, Max iterations
{
  // Initialize the MPI environment

  if (argc < 2)
  {
    printf("Usage: %s <your_argument>\n", argv[0]);
    return 1; // Return an error code
  }
  int numb = atoi(argv[1]);
  
  int size = (int)pow(2.0, numb);
  int array[(int)size];
  srand(time(NULL)); // random numbers into array
  for (int i = 0; i < size; i++)
  {
    array[i] = rand() % 100000; // Generate random numbers between 0 and 99999
  }

  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int *localSortedSegment;
  int *mergedArray = NULL;
  int segmentSize = world_size / size;

  MPI_Request *requests = NULL;
  MPI_Status *statuses = NULL;
  double numb = atof(argv[1]);
  int *size = (int)pow(2.0, numb);
  int array[(int)size];
  int tempArr;
  if (world_rank == 0)
  {

    srand(time(NULL)); // random numbers into array
    for (int i = 0; i < size; i++)
    {
      array[i] = rand() % 100000; // Generate random numbers between 0 and 99999
    }
    mergedArray = (int *)malloc(size * sizeof(int));
    requests = (MPI_Request *)malloc((size - 1) * sizeof(MPI_Request));
    statuses = (MPI_Status *)malloc((size - 1) * sizeof(MPI_Status));
    tempArr = (int *)malloc(size * sizeof(int)); // Allocate tempArr
  }

  // Broadcast array from rank 0 to all processes
  localSortedSegment = (int *)malloc(segmentSize * sizeof(int));
  MPI_Scatter(array, segmentSize, MPI_INT, localSortedSegment, segmentSize, MPI_INT, 0, MPI_COMM_WORLD);

  //sort 
  sort(localSortedSegment, segmentSize, true);


  if (world_rank != 0) {
        MPI_Send(localSortedSegment, segmentSize, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Copy local sorted segment directly to merged array
        memcpy(mergedArray, localSortedSegment, segmentSize * sizeof(int));
        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&mergedArray[i * segmentSize], segmentSize, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }
        // Wait for all receives to complete
        MPI_Waitall(world_size - 1, requests, statuses);
        // Merge sorted segments into merged array
        mergeSortedSegments(mergedArray, localSortedSegment, tempArr, size, segmentSize);
    }

    // Clean up
    free(array);
    free(localSortedSegment);
    if (world_rank == 0) {
        free(mergedArray);
        free(requests);
        free(statuses);
    }

    MPI_Finalize();
  // Print off a hello world message
  printf("Hello world from CPU land on processor %s, rank %d"
         " out of %d processors\n",
         processor_name, world_rank, world_size);

  // Call CUDA land from MPI
  runCudaLand(world_rank);

  // Finalize the MPI environment.
  MPI_Finalize();
}
