#include <chrono>
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "check.h"
#include <thread>


void sumArrays(const float *a, const float *b, float *res, const int size) {
    for (int i = 0; i < size; i+=4) {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}


__global__ void sumArraysGPU(const float *a, const float *b, float *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    res[idx] = a[idx] + b[idx];
}


int main() {
    bool pin = false;
    cudaSetDevice(0);

    int nElem = 1 << 14;
    printf("Elem size: %d \n", nElem);

    int bytes = sizeof(float) * nElem;

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_res = (float *)malloc(bytes);
    float *d_to_h_res = (float *)malloc(bytes);

    memset(h_a, 0, bytes);
    memset(h_b, 0, bytes);

    for (int i = 0; i < nElem; ++i) {
        h_a[i] = 2.34;
        h_b[i] = 2.36;
    }

    float *d_a, *d_b, *d_res;
    if (pin) {
        std::cout << "USE PINE MEMORY" << std::endl;
        CHECK(cudaMallocHost((float **)&d_a, bytes));
        CHECK(cudaMallocHost((float **)&d_b, bytes));
        CHECK(cudaMallocHost((float **)&d_res, bytes));
    }
    else {
        std::cout << "USE DYNAMIC MEMORY" << std::endl;
        CHECK(cudaMalloc((float **)&d_a, bytes));
        CHECK(cudaMalloc((float **)&d_b, bytes));
        CHECK(cudaMalloc((float **)&d_res, bytes));
    }

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem / block.x);
    printf("Execution configuration <<<%d, %d>>> \n", block.x, grid.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sumArraysGPU<<<block, grid>>>(d_a, d_b, d_res);
    cudaEventRecord(stop);

    CHECK(cudaMemcpy(d_to_h_res, d_res, bytes, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "GPU USE TIME: " << elapsed_time << "ms. " << std::endl;
    
    sumArrays(h_a, h_b, h_res, nElem);

    for (int i = 0; i < nElem; ++i) {
        if (abs(h_res[i] - d_to_h_res[i]) > 1e-8) {
            std::cout<< "ERROR" << std::endl;
            return 0;
        }
    }

    std::cout << "SUCCESS" << std::endl;
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_res);

    free(h_a);
    free(h_b);
    free(h_res);
    free(d_to_h_res);

    return 0;


}