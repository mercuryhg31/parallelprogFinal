#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
	int numranks = 9;
	if (sqrt(numranks) - (int) sqrt(numranks) != 0.0) {
		printf("NOT A SQUARE\n");
	}
	else printf("good\n");
	int *test = calloc((size * size), sizeof(int));
}
