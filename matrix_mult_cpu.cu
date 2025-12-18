#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE double
// функция перемножения матриц


void matrixMult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Brows, int Bcols) {

	for (int i = 0; i < Arows; i++) {
		for (int j = 0; j < Bcols; j++) {
			BASE_TYPE sum = 0;
			for (int k = 0; k < Acols; k++) {
				sum += A[i * Acols + k] * B[k * Bcols + j];
			}
			C[i * Bcols + j] = sum;
		}
	}
}

int toMultiple(int a, int b) {
	int mod = a % b;
	if (mod != 0) {
		mod = b - mod;
		return a + mod;
	}
	return a;
}
int main()
{
	//start, stop - for Kernel time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// количество строк и столбцов матрицы
	int Arows = 1000;
	int Acols = 2000;
	int Brows = Acols;
	int Bcols = 1500;

	Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);

	Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);

	Brows = toMultiple(Brows, BLOCK_SIZE);
	printf("Brows = %d\n", Brows);

	Bcols = toMultiple(Bcols, BLOCK_SIZE);
	printf("Bcols = %d\n", Bcols);

	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
	size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
	BASE_TYPE* h_A = (BASE_TYPE*)malloc(Asize);
	BASE_TYPE* h_B = (BASE_TYPE*)malloc(Bsize);
	BASE_TYPE* h_C = (BASE_TYPE*)malloc(Csize);

	for (int i = 0; i < Arows * Acols; ++i) {
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Brows * Bcols; ++i) {
		h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
	}

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	auto cpu_start = high_resolution_clock::now();
	matrixMult(h_A, h_B, h_C, Arows, Acols, Brows, Bcols);
	auto cpu_compute_end = high_resolution_clock::now();
	auto cpu_compute_time = duration_cast<milliseconds>(cpu_compute_end - cpu_start);
	printf("CPU Time: %lld ms\n", cpu_compute_time.count());

	printf("Test STARTED\n");
	for (int i = 0; i < Arows; i++) {
		for (int j = 0; j < Bcols; j++) {
			BASE_TYPE sum = 0;
			for (int k = 0; k < Acols; k++)
				sum += h_A[i * Acols + k] * h_B[k *
				Bcols + j];

			if (fabs(h_C[i * Bcols + j] - sum) > 1e-3)
			{
				fprintf(stderr, "Result verification failed at element[% d, % d]!\n", i, j);
				printf("sum = %f, h_C[i * Bcols + j] =% f\n", sum, h_C[i * Bcols + j]);
				exit(EXIT_FAILURE);
			}
		}
	}
	printf("Test PASSED\n");
	free(h_A);
	free(h_B);
	free(h_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}