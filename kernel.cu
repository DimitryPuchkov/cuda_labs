
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>

// Use stb for image IO
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#define GPU1  0
#define GPU2  1

const int CHANNELS = 3;


// Simple helper to clamp coordinates
__device__ __host__ inline int clamp(int v, int a, int b) { return v < a ? a : (v > b ? b : v); }

// helper: load center + halo into shared memory
__device__ __forceinline__ void load_shared_for_tile(const unsigned char* in, unsigned char* sdata,
    int width, int height,
    int tx, int ty, int bx, int by, int sWidth, int sHeight)
{
    int x = bx + tx;
    int y = by + ty;

    // center
    for (int c = 0; c < CHANNELS; ++c) {
        int s_x = tx + 1;
        int s_y = ty + 1;
        int sIdx = (s_y * sWidth + s_x) * CHANNELS + c;
        int imgX = clamp(x, 0, width - 1);
        int imgY = clamp(y, 0, height - 1);
        if (x < width && y < height) sdata[sIdx] = in[(imgY * width + imgX) * CHANNELS + c];
        else sdata[sIdx] = 0;
    }

    // left halo
    if (tx == 0) {
        int lx = bx + tx - 1; int loadX = clamp(lx, 0, width - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = ((ty + 1) * sWidth + 0) * CHANNELS + c;
            int imgY = clamp(y, 0, height - 1);
            if (y < height) sdata[sIdx] = in[(imgY * width + loadX) * CHANNELS + c]; else sdata[sIdx] = 0;
        }
    }
    // right halo
    if (tx == blockDim.x - 1) {
        int rx = bx + tx + 1; int loadX = clamp(rx, 0, width - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = ((ty + 1) * sWidth + (sWidth - 1)) * CHANNELS + c;
            int imgY = clamp(y, 0, height - 1);
            if (y < height) sdata[sIdx] = in[(imgY * width + loadX) * CHANNELS + c]; else sdata[sIdx] = 0;
        }
    }
    // top halo
    if (ty == 0) {
        int uy = by + ty - 1; int loadY = clamp(uy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = (0 * sWidth + (tx + 1)) * CHANNELS + c;
            int imgX = clamp(x, 0, width - 1);
            if (x < width) sdata[sIdx] = in[(loadY * width + imgX) * CHANNELS + c]; else sdata[sIdx] = 0;
        }
    }
    // bottom halo
    if (ty == blockDim.y - 1) {
        int dy = by + ty + 1; int loadY = clamp(dy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = ((sHeight - 1) * sWidth + (tx + 1)) * CHANNELS + c;
            int imgX = clamp(x, 0, width - 1);
            if (x < width) sdata[sIdx] = in[(loadY * width + imgX) * CHANNELS + c]; else sdata[sIdx] = 0;
        }
    }

    // corners
    if (tx == 0 && ty == 0) {
        int cx = bx + tx - 1; int cy = by + ty - 1;
        int lx = clamp(cx, 0, width - 1); int ly = clamp(cy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = (0 * sWidth + 0) * CHANNELS + c;
            sdata[sIdx] = in[(ly * width + lx) * CHANNELS + c];
        }
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        int cx = bx + tx - 1; int cy = by + ty + 1;
        int lx = clamp(cx, 0, width - 1); int ly = clamp(cy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = ((sHeight - 1) * sWidth + 0) * CHANNELS + c;
            sdata[sIdx] = in[(ly * width + lx) * CHANNELS + c];
        }
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        int cx = bx + tx + 1; int cy = by + ty - 1;
        int lx = clamp(cx, 0, width - 1); int ly = clamp(cy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = (0 * sWidth + (sWidth - 1)) * CHANNELS + c;
            sdata[sIdx] = in[(ly * width + lx) * CHANNELS + c];
        }
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int cx = bx + tx + 1; int cy = by + ty + 1;
        int lx = clamp(cx, 0, width - 1); int ly = clamp(cy, 0, height - 1);
        for (int c = 0; c < CHANNELS; ++c) {
            int sIdx = ((sHeight - 1) * sWidth + (sWidth - 1)) * CHANNELS + c;
            sdata[sIdx] = in[(ly * width + lx) * CHANNELS + c];
        }
    }
}

// Shared-memory 3x3 processing kernel (modified to use helper)

__global__ void blur_shared_kernel(const unsigned char* in, unsigned char* out, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    extern __shared__ unsigned char sdata[]; // (blockDim.x+2)*(blockDim.y+2)*CHANNELS
    int sWidth = blockDim.x + 2;
    int sHeight = blockDim.y + 2;

    // fill shared tile (center + halo)
    load_shared_for_tile(in, sdata, width, height, tx, ty, bx, by, sWidth, sHeight);

    __syncthreads();

    if (x >= width || y >= height) return;
    float kernel[3][3] = {
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
    };

    for (int c = 0; c < CHANNELS; ++c) {
        float sum = 0.0f;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                int s_x = (tx + 1) + ox;
                int s_y = (ty + 1) + oy;
                unsigned char value = sdata[(s_y * sWidth + s_x) * CHANNELS + c];
                sum += kernel[oy + 1][ox + 1] * (float)value;
            }
        }
        int result_value = (int)(sum + 0.5f);
        if (result_value < 0) result_value = 0; else if (result_value > 255) result_value = 255;
        out[(y * width + x) * CHANNELS + c] = (unsigned char)result_value;
    }
}

// Shared Sobel
__global__ void sobel_shared_kernel(const unsigned char* in, unsigned char* out, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    extern __shared__ unsigned char sdata[]; // (blockDim.x+2)*(blockDim.y+2)*CHANNELS
    int sWidth = blockDim.x + 2;
    int sHeight = blockDim.y + 2;

    // fill shared tile (center + halo)
    load_shared_for_tile(in, sdata, width, height, tx, ty, bx, by, sWidth, sHeight);

    __syncthreads();

    if (x >= width || y >= height) return;
    const int kernel_x[3][3] = {
        { -1,0,1 },
        { -2,0,2 },
        { -1,0,1 }
    };
    const int kernel_y[3][3] = {
        { 1,2,1 },
        { 0,0,0 },
        { -1,-2,-1 }
    };
    for (int c = 0; c < CHANNELS; ++c) {
        int sum_x = 0, sum_y = 0;
        for (int oy = -1; oy <= 1; ++oy)
            for (int ox = -1; ox <= 1; ++ox) {
                int s_x = (tx + 1) + ox;
                int s_y = (ty + 1) + oy;
                int value = sdata[(s_y * sWidth + s_x) * CHANNELS + c];
                sum_x += kernel_x[oy + 1][ox + 1] * value;
                sum_y += kernel_y[oy + 1][ox + 1] * value;
            }
        int mag = abs(sum_x) + abs(sum_y);
        if (mag > 255) mag = 255;
        out[(y * width + x) * CHANNELS + c] = (unsigned char)mag;
    }
}


int main(int argc, char* argv[])
{
    using clock = std::chrono::high_resolution_clock;
    auto total_start = clock::now();

    const char* inPath = argv[1];
    const char* outPath = argv[2];
    const char* filter = argv[3];

    // чтение изображения
    int width = 0, height = 0, channels = 0;
    // в image будет лежать массив пикселей, на каждый пиксель отводится channelsбайт, пикселей width*height
    // формат пикселей  [RGB, RGB, RGB...], строки идут сверху вниз, первые 3 байта - цвет верхнего левого пикселя, вторые 3 байта - цвет второго пикселя в верхней строке 
    // байты width*channels, width*channels+1, width*channels+2 - цвет первого пикселя второй строки и т.д.
    unsigned char* image = stbi_load(inPath, &width, &height, &channels, 0);
    if (!image) { std::cerr << "Failed to load " << inPath << std::endl; return 1; }
    std::cout << "Loaded " << inPath << ": " << width << "x" << height << " channels=" << channels << std::endl;

    size_t numBytes = (size_t)width * height * channels;
    // Разделяем изображение на две части с перекрытием (halo)
    int half_height = height / 2;
    int overlap = 1; // Для фильтра 3x3 нужен 1 пиксель перекрытия

    // Размеры частей с учетом перекрытия
    int part1_height = half_height + overlap;
    int part2_height = height - half_height + overlap;

    size_t image1Size = (size_t)width * part1_height * channels;
    size_t image2Size = (size_t)width * part2_height * channels;

    // Выделяем память для частей
    unsigned char* image1 = new unsigned char[image1Size];
    unsigned char* image2 = new unsigned char[image2Size];

    // разделение изображения на 2 части с учетом halo
    memcpy(image1, image, image1Size);
    memcpy(image2, image + image1Size - (size_t)width * 2 * channels, image2Size);

    cudaDeviceSynchronize();
    cudaSetDevice(GPU1);
    unsigned char* d_in1 = nullptr; // массив пикселей на устройстве для входного изображения на 1 девайсе
    unsigned char* d_out1 = nullptr; // массив пикселей на устройстве для выходного изображения на 1 девайсе
    // выделение памяти на устройстве
    cudaError_t err;
    err = cudaMalloc((void**)&d_in1, image1Size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc in failed: " << cudaGetErrorString(err) << std::endl; stbi_image_free(image); return 1; }
    err = cudaMalloc((void**)&d_out1, image1Size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc out failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in1); stbi_image_free(image); return 1; }


    cudaSetDevice(GPU2);
    unsigned char* d_in2 = nullptr; // массив пикселей на устройстве для входного изображения на 2 девайсе
    unsigned char* d_out2 = nullptr; // массив пикселей на устройстве для выходного изображения на 2 девайсе
    // выделение памяти на устройстве
    err = cudaMalloc((void**)&d_in2, image2Size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc in failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in1); cudaFree(d_out1); stbi_image_free(image); return 1; }
    err = cudaMalloc((void**)&d_out2, image2Size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc out failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in1); cudaFree(d_out1); cudaFree(d_in2); stbi_image_free(image); return 1; }

    // размер блока 16x16 так как удобно для паралелизма (кратно 32) и достаточно по размеру для загрузки соседних пикселей в shared память
    dim3 block(16, 16);
    
    size_t sharedBytes = (block.x + 2) * (block.y + 2) * channels; // shared memory по размеру на 2 пикселя больше блока в каждую сторону (для соседних пикселей)
    cudaEvent_t start, stop;

    cudaSetDevice(GPU1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float kernel_ms1 = 0.0f;
    float h2d_ms1 = 0.0f;
    float d2h_ms1 = 0.0f;



    // копирование входного изображения на устройство
    cudaEventRecord(start);
    err = cudaMemcpy(d_in1, image1, image1Size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // размер сетки расчитывается исходя из размера изображения и размера блока (гарантируем покрытие всего изображения)
    dim3 grid1((width + block.x - 1) / block.x, (part1_height + block.y - 1) / block.y);
    cudaEventElapsedTime(&h2d_ms1, start, stop);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in1); cudaFree(d_out1); stbi_image_free(image); return 1; }

    // запуск ядра в зависимости от выбранного фильтра
    cudaEventRecord(start);
    if (strcmp(filter, "blur") == 0) {
        blur_shared_kernel << <grid1, block, (unsigned int)sharedBytes >> > (d_in1, d_out1, width, part1_height);
    }
    else if (strcmp(filter, "sobel") == 0) {
        sobel_shared_kernel << <grid1, block, (unsigned int)sharedBytes >> > (d_in1, d_out1, width, part1_height);
    }
    else {
        std::cerr << "Unknown filter " << filter << std::endl;
        cudaFree(d_in1); cudaFree(d_out1); stbi_image_free(image); return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_ms1, start, stop);

    // копирование результата на хост
    cudaDeviceSynchronize();
    unsigned char* outHost1 = new unsigned char[image1Size];
    cudaEventRecord(start);
    err = cudaMemcpy(outHost1, d_out1, (size_t)width * half_height * channels, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_ms1, start, stop);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in1); cudaFree(d_out1); stbi_image_free(image); free(outHost1); return 1; }



    cudaSetDevice(GPU2);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float kernel_ms2 = 0.0f;
    float h2d_ms2 = 0.0f;
    float d2h_ms2 = 0.0f;



    // копирование входного изображения на устройство
    cudaEventRecord(start);
    err = cudaMemcpy(d_in2, image2, image2Size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_ms2, start, stop);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in2); cudaFree(d_out2); stbi_image_free(image); return 1; }


    // запуск ядра в зависимости от выбранного фильтра
    dim3 grid2((width + block.x - 1) / block.x, (part2_height + block.y - 1) / block.y);
    cudaEventRecord(start);
    if (strcmp(filter, "blur") == 0) {
        blur_shared_kernel << <grid2, block, (unsigned int)sharedBytes >> > (d_in2, d_out2, width, part2_height);
    }
    else if (strcmp(filter, "sobel") == 0) {
        sobel_shared_kernel << <grid2, block, (unsigned int)sharedBytes >> > (d_in2, d_out2, width, part2_height);
    }
    else {
        std::cerr << "Unknown filter " << filter << std::endl;
        cudaFree(d_in2); cudaFree(d_out1); stbi_image_free(image); return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_ms2, start, stop);

    // копирование результата на хост
    cudaDeviceSynchronize();
    unsigned char* outHost2 = new unsigned char[image2Size];
    cudaEventRecord(start);
    err = cudaMemcpy(outHost2, d_out2, (size_t)width * (height - half_height) * channels, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_ms2, start, stop);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_in2); cudaFree(d_out2); stbi_image_free(image); free(outHost2); return 1; }


    unsigned char* outHost = (unsigned char*)malloc(numBytes);
    // объединение двух частей обратно в одно изображение
    // разделение изображения на 2 части с учетом halo
    memcpy(outHost, outHost1, image1Size - width * channels);
    memcpy(outHost + image1Size - width * channels, outHost2, image2Size - width * channels);

    // сохранение результата
    int saved = stbi_write_png(outPath, width, height, channels, outHost, width * channels);
    if (!saved) std::cerr << "Failed to write " << outPath << std::endl; else std::cout << "Saved " << outPath << std::endl;

    // освобождение памяти
    cudaSetDevice(GPU1);
    cudaFree(d_in1); cudaFree(d_out1);
    cudaSetDevice(GPU2);
    cudaFree(d_in2); cudaFree(d_out2);
    stbi_image_free(image); free(outHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    // вывод времени выполнения
    auto total_end = clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    std::cout.precision(4);
    std::cout << "1)Host-to-Device copy time (ms): " << h2d_ms1 << std::endl;
    std::cout << "1)Kernel time (ms): " << kernel_ms1 << std::endl;
    std::cout << "1)Device-to-Host copy time (ms): " << d2h_ms1 << std::endl;

    std::cout << "2)Host-to-Device copy time (ms): " << h2d_ms2 << std::endl;
    std::cout << "2)Kernel time (ms): " << kernel_ms2 << std::endl;
    std::cout << "2)Device-to-Host copy time (ms): " << d2h_ms2 << std::endl;

    std::cout << "Total time (ms): " << total_ms << std::endl;

    return 0;
}
