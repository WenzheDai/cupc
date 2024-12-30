#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <random>
#include "check.h"

void sumArrays(float *a, float *b, float *res, const int size) {
    for (int i = 0; i < size; ++i) {
        res[i] = a[i] + b[i];
    }
}


__global__ void sumArrayGPU(float *a, float *b, float *res) {
    int i = threadIdx.x;
    res[i] = a[i] + b[i];
}

void initData(float *ip, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(1, 100);
    for (int i = 0; i < size; ++i) {
        ip[i] = static_cast<float>(distrib(gen));
    }
}

void checkResult(float * hostRef,float * gpuRef,const int N)
{
    double epsilon=1.0E-8;
    for(int i=0; i<N; ++i)
    {
        if(abs(hostRef[i]-gpuRef[i])>epsilon)
        {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
            return;
        }
    }
    printf("Check result success!\n");
}


int main() {
    int device = 0;
    cudaSetDevice(device);

    int nElem = 32;
    printf("Vector size: %d \n", nElem);
    int nByte = sizeof(float) * nElem;

    float *a_h = (float *)malloc(nByte);
    float *b_h = (float *)malloc(nByte);
    float *res_h = (float *)malloc(nByte);
    float *res_from_gpu_h = (float *)malloc(nByte);

    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float **)&a_d, nByte));
    CHECK(cudaMalloc((float **)&b_d, nByte));
    CHECK(cudaMalloc((float **)&res_d, nByte));

    initData(a_h, nElem);
    initData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(nElem);
    dim3 grid(nElem/block.x);

    sumArrayGPU<<<grid, block>>>(a_d, b_d, res_d);
    CHECK(cudaDeviceSynchronize());
    printf("Execution configuration <<<%d, %d>>> \n", block.x, grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);
    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}


