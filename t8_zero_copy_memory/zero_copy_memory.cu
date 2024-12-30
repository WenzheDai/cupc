#include <chrono>
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "check.h"


void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}


__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  res[i]=a[i]+b[i];
}

int main() {
    cudaSetDevice(0);
    int nElem = 1 << 10;
    printf("Vector size : %d \n", nElem);

    int bytes = nElem * sizeof(float);
    float *res_from_gpu_h = (float *)malloc(bytes);
    float *h_res = (float *)malloc(bytes);

    memset(res_from_gpu_h, 0, bytes);
    memset(h_res, 0, bytes);

    float *a_h, *b_h, *res_d;
    dim3 block(1024);
    dim3 grid(nElem / block.x);

    res_from_gpu_h = (float *)malloc(bytes);
    float *d_a, *d_b;

    CHECK(cudaHostAlloc((float **)&a_h, bytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((float **)&b_h, bytes, cudaHostAllocMapped));
    CHECK(cudaMalloc((float **)&res_d, bytes));

    for (int i = 0; i < nElem; ++i) {
        a_h[i] = 2.34;
        b_h[i] = 2.56;
    }

    // ===================================================================================//
    std::cout << "==================zero copy memory========================" << std::endl;
    // 此方式已经被 UVA淘汰

    auto start_cp = std::chrono::high_resolution_clock::now();
    CHECK(cudaHostGetDevicePointer((void **)&d_a, (void **) a_h, 0));
    CHECK(cudaHostGetDevicePointer((void **)&d_b, (void **) b_h, 0));
    sumArraysGPU<<<grid, block>>>(d_a, d_b, res_d);
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, bytes, cudaMemcpyDeviceToHost));
    auto end_cp = std::chrono::high_resolution_clock::now();
    auto elapsed_time_cp = std::chrono::duration_cast<std::chrono::microseconds>(end_cp - start_cp);
    std::cout << "ZERO COPY MEMORY USE TIME: " << elapsed_time_cp.count() << "us" << std::endl;

    std::cout << "==================normal memory========================" << std::endl;
    float *a_h_n = (float *)malloc(bytes);
    float *b_h_n = (float *)malloc(bytes);
    float *res_h_n = (float *)malloc(bytes);
    float *res_from_gpu_h_n = (float *)malloc(bytes);

    memset(a_h_n, 0, bytes);
    memset(b_h_n, 0, bytes);

    for (int i = 0; i < nElem; ++i) {
        a_h_n[i] = 2.34;
        b_h_n[i] = 2.56;
    }

    cudaEvent_t start_n, end_n;
    cudaEventCreate(&start_n);
    cudaEventCreate(&end_n);

    float *d_a_n, *d_b_n, *d_res_n;

    cudaEventRecord(start_n);

    CHECK(cudaMalloc((float **)&d_a_n, bytes));
    CHECK(cudaMalloc((float **)&d_b_n, bytes));
    CHECK(cudaMalloc((float **)&d_res_n, bytes));

    CHECK(cudaMemcpy(d_a_n, a_h_n, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_n, b_h_n, bytes, cudaMemcpyHostToDevice));

    sumArraysGPU<<<grid, block>>>(d_a_n, d_b_n, d_res_n);
    CHECK(cudaMemcpy(res_from_gpu_h_n, d_res_n, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(end_n);

    cudaEventSynchronize(end_n);
    float elasped_time_n = 0;
    cudaEventElapsedTime(&elasped_time_n, start_n, end_n);
    std::cout << "NORMAL USE TIEM: " << elasped_time_n * 1000 << "us" << std::endl;

    sumArrays(a_h_n, b_h_n, res_h_n, nElem);

    for (int i = 0; i < nElem; ++i) {
        if (abs(res_h_n[i] - res_from_gpu_h[i]) > 1e-8) {
            std::cout<< "ZERO COPY MEMPCY ERROR" << std::endl;
            return 0;
        }

        if (abs(res_h_n[i] - res_from_gpu_h_n[i]) > 1e-8) {
            std::cout<< "NORMAL MEMPCY ERROR" << std::endl;
            return 0;
        }
    }

    std::cout << "==================Result========================" << std::endl;
    std::cout << "SUCCESS" << std::endl;

    cudaFreeHost(a_h);
    cudaFreeHost(b_h);
    cudaFree(d_a_n);
    cudaFree(d_b_n);
    cudaFree(d_res_n);
    cudaFree(res_d);

    free(a_h_n);
    free(b_h_n);
    free(h_res);
    free(res_from_gpu_h);
    free(res_from_gpu_h_n);

    return 0;

}