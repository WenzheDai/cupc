#include <stdio.h>
#include <iostream>
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
    int nElem = 1 << 14;
    printf("Vector size:%d\n",nElem);

    int bytes = sizeof(float) * nElem;

    float *h_a, *h_b, *h_from_gpu;
    h_from_gpu = (float *)malloc(bytes);

    float *d_res;

    CHECK(cudaHostAlloc((void **)&h_a, bytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&h_b, bytes, cudaHostAllocMapped));
    CHECK(cudaMalloc((float **)&d_res, bytes));

    for (int i = 0; i < nElem; ++i) {
        h_a[i] = 2.37;
        h_b[i] = 2.33;
    }

    dim3 block(1024);
    dim3 grid(nElem / 1024);

    // 使用UVA,不再需要 cudaHostGetDevicePointor 方法
    sumArraysGPU<<<grid, block>>>(h_a, h_b, d_res);
    CHECK(cudaMemcpy(h_from_gpu, d_res, bytes, cudaMemcpyDeviceToHost));

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *res = (float *)malloc(bytes);


    for (int i = 0; i < nElem; ++i) {
        a[i] = 2.37;
        b[i] = 2.33;
    }
    sumArrays(a, b, res, nElem);

    for (int i = 0; i < nElem; ++i) {
        if (abs(res[i] - h_from_gpu[i]) > 1e-8) {
            std::cout<< "ZERO COPY MEMPCY ERROR" << std::endl;
            return 0;
        }
    }
    std::cout << "SUCCESS" << std::endl;

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
    free(h_from_gpu);

    return 0;
}
