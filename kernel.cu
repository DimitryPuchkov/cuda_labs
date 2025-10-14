#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <math_functions.h> 
#define N 1000000000
#define PI 3.14159265358979323846f
#include <iostream>
#include <string>
#include <iomanip>
using namespace std;

#define dtype float

__global__ void init_arr_sin(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = sin((i % 360) * PI / (dtype)180.0);
    }
}

__global__ void init_arr_sinf(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = sinf((i % 360) * PI / (dtype)180.0);
    }
}

__global__ void init_arr_cuda__sinf(dtype* arr, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = __sinf((i % 360) * PI / (dtype)180.0);
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");
    int device = 0;
    dim3 blockSize(1024);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    dtype* dev_arr;
    dtype* arr = nullptr;
    double err = 0.0;
    cudaDeviceProp prop;
    cudaError_t cuda_error;

    cudaSetDevice(device);

    cuda_error = cudaGetDeviceProperties(&prop, device);
    if (cuda_error != cudaSuccess) {
        cout << "Ошибка получения свойств устройства: " << cudaGetErrorString(cuda_error) << endl;
        return 1;
    }

    cout << "Имя устройства: " << prop.name << endl;
    cout << "Количество мультипроцессоров: " << prop.multiProcessorCount << endl;
    cout << "Объем глобальной памяти: " << prop.totalGlobalMem << " байт" << endl;
    cout << "Максимальное количество потоков на блок: " << prop.maxThreadsPerBlock << endl;
    cout << "Максимальный размер сетки: " 
         << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "Максимальный размер блока: " 
         << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;

    cudaMalloc((void**)&dev_arr, N * sizeof(dtype));
    arr = (dtype*)malloc(N * sizeof(dtype));

    // --- Время для init_arr_sin ---
    cudaEvent_t start_sin, stop_sin;
    cudaEventCreate(&start_sin);
    cudaEventCreate(&stop_sin);
    cudaEventRecord(start_sin, 0);

    init_arr_sin<<<gridSize, blockSize>>>(dev_arr, N);
    cudaEventRecord(stop_sin, 0);
    cudaEventSynchronize(stop_sin);

    float ms_sin = 0.0f;
    cudaEventElapsedTime(&ms_sin, start_sin, stop_sin);

    cudaMemcpy(arr, dev_arr, N * sizeof(dtype), cudaMemcpyDeviceToHost);
    err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        dtype expected = sin((i % 360) * PI / (dtype)180.0);
        err += fabs(expected - arr[i]);
    }
    err /= N;
    cout << "sin err = " << std::setprecision(10) << std::scientific << err << endl;
    cout << "Время выполнения init_arr_sin: " << ms_sin << " мс" << endl;

    cudaEventDestroy(start_sin);
    cudaEventDestroy(stop_sin);

    // --- Время для init_arr_sinf ---
    cudaEvent_t start_sinf, stop_sinf;
    cudaEventCreate(&start_sinf);
    cudaEventCreate(&stop_sinf);
    cudaEventRecord(start_sinf, 0);

    init_arr_sinf<<<gridSize, blockSize>>>(dev_arr, N);
    cudaEventRecord(stop_sinf, 0);
    cudaEventSynchronize(stop_sinf);

    float ms_sinf = 0.0f;
    cudaEventElapsedTime(&ms_sinf, start_sinf, stop_sinf);

    cudaMemcpy(arr, dev_arr, N * sizeof(dtype), cudaMemcpyDeviceToHost);
    err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        dtype expected = sin((i % 360) * PI / (dtype)180.0);
        err += fabs(expected - arr[i]);
    }
    err /= N;
    cout << "sinf err = " << std::setprecision(10) << std::scientific << err << endl;
    cout << "Время выполнения init_arr_sinf: " << ms_sinf << " мс" << endl;
    
    cudaEventDestroy(start_sinf);
    cudaEventDestroy(stop_sinf);

    // --- Время для init_arr_cuda__sinf ---
    cudaEvent_t start_cuda_sinf, stop_cuda_sinf;
    cudaEventCreate(&start_cuda_sinf);
    cudaEventCreate(&stop_cuda_sinf);
    cudaEventRecord(start_cuda_sinf, 0);

    init_arr_cuda__sinf<<<gridSize, blockSize>>>(dev_arr, N);
    cudaEventRecord(stop_cuda_sinf, 0);
    cudaEventSynchronize(stop_cuda_sinf);

    float ms_cuda_sinf = 0.0f;
    cudaEventElapsedTime(&ms_cuda_sinf, start_cuda_sinf, stop_cuda_sinf);

    cudaMemcpy(arr, dev_arr, N * sizeof(dtype), cudaMemcpyDeviceToHost);
    err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        dtype expected = sin((i % 360) * PI / (dtype)180.0);
        err += fabs(expected - arr[i]);
    }
    err /= N;
    cout << "__sinf err = " << std::setprecision(10) << std::scientific << err << endl;
    cout << "Время выполнения init_arr_cuda__sinf: " << ms_cuda_sinf << " мс" << endl;

    cudaEventDestroy(start_cuda_sinf);
    cudaEventDestroy(stop_cuda_sinf);

    cudaFree(dev_arr);
    free(arr);
    return 0;
}