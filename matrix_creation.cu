#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <locale.h>

// Ядро для создания матрицы
__global__ void createMatrix(int* A, const int n)
{
    // Создание элементов матрицы на GPU
    A[threadIdx.y * n + threadIdx.x] = 10 *
        threadIdx.y + threadIdx.x;
}

// Функция создания матрицы на CPU
void createMatrixCPU(int* A, const int n)
{
    // Создание элементов матрицы на CPU
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = 10 * i + j;
}

int main()
{
    setlocale(LC_ALL, "Russian");
    // Инициализация переменных для измерения времени
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    // кол-во строк и столбцов матрицы
    const int n = 10; 
    // размер матрицы
    size_t size = n * n * sizeof(int);
    printf("\n=== CPU Время выполнения ===\n");

    // Выделяем память для матрицы на CPU
    auto cpu_start = high_resolution_clock::now();
    int* h_A = (int*)malloc(size);
    if (!h_A) {
        printf("Ошибка выделения памяти на CPU!\n");
        return 1;
    }
    auto cpu_malloc_end = high_resolution_clock::now();

    // Создание матрицы на CPU
    createMatrixCPU(h_A, n);
    auto cpu_compute_end = high_resolution_clock::now();

    // Расчет времени на CPU
    auto cpu_malloc_time = duration_cast<microseconds>(cpu_malloc_end - cpu_start);
    auto cpu_compute_time = duration_cast<microseconds>(cpu_compute_end - cpu_malloc_end);
    auto cpu_total_time = duration_cast<microseconds>(cpu_compute_end - cpu_start);

    printf("Выделение памяти: %lld мкс (%.3f мс)\n",
        cpu_malloc_time.count(), cpu_malloc_time.count() / 1000.0);
    printf("Заполнение матрицы: %lld мкс (%.3f мс)\n",
        cpu_compute_time.count(), cpu_compute_time.count() / 1000.0);
    printf("Общее время CPU: %lld мкс (%.3f мс)\n",
        cpu_total_time.count(), cpu_total_time.count() / 1000.0);




    printf("\n=== GPU Время выполнения ===\n");
    cudaError_t cudaStatus;
    cudaEvent_t start, stop, start_malloc, stop_malloc, start_compute, stop_compute, start_copy, stop_copy;
    float gpu_malloc_ms = 0.0f, gpu_compute_ms = 0.0f,
        gpu_copy_ms = 0.0f, gpu_total_ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);

    // Общее время начала работы с GPU
    cudaEventRecord(start);

    // ---- 1. Выделение памяти на GPU ----
    cudaEventRecord(start_malloc);
    int* d_B = NULL;
    cudaStatus = cudaMalloc((void**)&d_B, size);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        free(h_A);
        return 1;
    }
    cudaEventRecord(stop_malloc);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&gpu_malloc_ms, start_malloc, stop_malloc);

    // ---- 2. Создание матрицы на GPU ----
    cudaEventRecord(start_compute);

    // Определение размеров сетки и блоков
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Конфигурация GPU: %dx%d блоков по %dx%d потоков\n",
        blocksPerGrid.x, blocksPerGrid.y,
        threadsPerBlock.x, threadsPerBlock.y);

    // Вызов ядра
    createMatrix << <blocksPerGrid, threadsPerBlock >> > (d_B, n);

    // Проверяем ошибки при запуске ядра
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_B);
        free(h_A);
        return 1;
    }

    cudaEventRecord(stop_compute);
    cudaEventSynchronize(stop_compute);
    cudaEventElapsedTime(&gpu_compute_ms, start_compute, stop_compute);

    // ---- 3. Копирование данных с GPU на хост ----
    cudaEventRecord(start_copy);

    // Выделяем память для матрицы B на хосте
    int* h_B = (int*)malloc(size);
    if (!h_B) {
        printf("Ошибка выделения памяти для h_B на CPU!\n");
        cudaFree(d_B);
        free(h_A);
        return 1;
    }

    // Копируем матрицу из GPU на CPU
    cudaStatus = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_B);
        free(h_A);
        free(h_B);
        return 1;
    }

    cudaEventRecord(stop_copy);
    cudaEventSynchronize(stop_copy);
    cudaEventElapsedTime(&gpu_copy_ms, start_copy, stop_copy);

    // Общее время работы GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_total_ms, start, stop);

    // ---- 4. Проверка результатов ----

    bool matrices_match = true;
    int error_count = 0;
    const int max_errors_to_print = 10;

    // Проверяем совпадение матрицы А и матрицы В
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (h_A[idx] != h_B[idx]) {
                matrices_match = false;
                error_count++;
                if (error_count <= max_errors_to_print) {
                    printf("Ошибка: h_A[%d][%d] = %d != h_B[%d][%d] = %d\n",
                        i, j, h_A[idx], i, j, h_B[idx]);
                }
                if (error_count == max_errors_to_print) {
                    printf("... и еще ошибок (всего %d)\n", error_count);
                }
            }
        }
    }

    // ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
    printf("\n--- Результаты замеров времени GPU ---\n");
    printf("1. Выделение памяти на GPU: %.3f мс\n", gpu_malloc_ms);
    printf("2. Заполнение матрицы на GPU: %.3f мс\n", gpu_compute_ms);
    printf("3. Копирование данных на хост: %.3f мс\n", gpu_copy_ms);
    printf("4. Общее время GPU: %.3f мс\n", gpu_total_ms);

    printf("\n--- Сравнение CPU vs GPU ---\n");
    printf("Общее время CPU: %.3f мс\n", cpu_total_time.count() / 1000.0);
    printf("Общее время GPU (с копированием): %.3f мс\n", gpu_total_ms);
    printf("Ускорение GPU (только вычисления): %.2fx\n",
        cpu_compute_time.count() / 1000.0 / gpu_compute_ms);
    printf("Ускорение GPU (с учетом передачи): %.2fx\n",
        cpu_total_time.count() / 1000.0 / gpu_total_ms);

    // Уничтожаем события CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);
    cudaEventDestroy(start_copy);
    cudaEventDestroy(stop_copy);

    // Освобождаем память на GPU
    cudaFree(d_B);

    // Освобождаем память на CPU
    free(h_A);
    free(h_B);

    // Сбрасываем устройство
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}