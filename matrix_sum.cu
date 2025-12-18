#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
// размер блока
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE float
// Ядро
// Функция транспонирования матрицы
__global__ void matrixSum(const BASE_TYPE* A, const
	BASE_TYPE* B, BASE_TYPE* C, int cols)
{
	// Индекс элемента в матрице
	int ind = cols * (blockDim.y * blockIdx.y +
		threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[ind] = A[ind] + B[ind];
}

// Функция вычисления числа, которое больше числа а
// и кратное числу b
int toMultiple(int a, int b)
{
	int mod = a % b;
	if (mod != 0)
	{
		mod = b - mod;
		return a + mod;
	}
	return a;
}
int main()
{
	// Объекты событий
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Количество строк и столбцов матрицы
	int rows = 1000;
	int cols = 2000;
	// Меняем количество строк и столбцов матрицы
	// на число, кратное размеру блока (16)
	rows = toMultiple(rows, BLOCK_SIZE);
	printf("rows = %d\n", rows);
	cols = toMultiple(cols, BLOCK_SIZE);
	printf("cols = %d\n", cols);
	size_t size = rows * cols * sizeof(BASE_TYPE);
	// Выделение памяти под матрицы на хосте
	BASE_TYPE* A = (BASE_TYPE*)malloc(size);
	BASE_TYPE* B = (BASE_TYPE*)malloc(size);
	BASE_TYPE* C = (BASE_TYPE*)malloc(size);
		// Инициализация матрицы
		for (int i = 0; i < rows * cols; ++i)
		{
			A[i] = rand() / (BASE_TYPE)RAND_MAX;
			B[i] = rand() / (BASE_TYPE)RAND_MAX;
		}
	// Выделение глобальной памяти на девайсе
	// для исходной матрицы
	BASE_TYPE* d_A = NULL;
	cudaMalloc((void**)&d_A, size);
	BASE_TYPE* d_B = NULL;
	cudaMalloc((void**)&d_B, size);
	BASE_TYPE* d_C = NULL;
	cudaMalloc((void**)&d_C, size);
	// Копируем матрицу из CPU на GPU
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE,
		BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE,
		rows / BLOCK_SIZE);
	// Начать отсчета времени
	cudaEventRecord(start, 0);
	// Запуск ядра
	matrixSum << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, cols);
	// Окончание работы ядра, остановка времени
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
		printf("KernelTime: %.2f milliseconds\n", KernelTime);
	// Копируем матрицу из GPU на CPU
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	// Проверка правильности работы ядра
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			if (A[i * cols + j]+B[i*cols+j] != C[i*cols+j])
				fprintf(stderr, "Result verification failed at element[% d, % d]!\n", i, j);
					exit(EXIT_FAILURE);
		}
	printf("Test PASSED\n");
	// Освобождаем память на GPU
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Освобождаем память на CPU
	free(A);
	free(B);
	free(C);
	// Удаляем объекты событий
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Done\n");
	return 0;
}