#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "check.h"


__global__ void sum_matrix(const float *MatA, const float *MatB, float *MatC) {
    // 2d-block  2d-thread
    int threadID_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int blockID_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadID_2D + blockDim.x * blockDim.y * blockID_2D;
    MatC[tid] = MatA[tid] + MatB[tid];
}


int main() {
    cudaSetDevice(0);
    int nx = 512;
    int ny = 512;

    int nbytes = sizeof(float) * nx * ny;

    float *h_matA = (float *)malloc(nbytes);
    float *h_matB = (float *)malloc(nbytes);
    float *h_matC = (float *)malloc(nbytes);

    for (int i = 0; i < nx * ny; ++i) {
        h_matA[i] = 2.8;
        h_matB[i] = 2.2;
    }
    
    float *d_matA, *d_matB, *d_matC;
    CHECK(cudaMalloc((float **)&d_matA, nbytes));
    CHECK(cudaMalloc((float **)&d_matB, nbytes));
    CHECK(cudaMalloc((float **)&d_matC, nbytes));

    CHECK(cudaMemcpy(d_matA, h_matA, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matB, h_matB, nbytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid(16, 16);
    sum_matrix<<<grid, block>>>(d_matA, d_matB, d_matC);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_matC, d_matC, nbytes, cudaMemcpyDeviceToHost));

    bool flag = true;
    for (int i = 0; i < nx * ny; ++i) {
        if (!(h_matC[i] - 5.0) < 1e-8) {
            std::cout << "no pass" << std::endl;
            std::cout << "result: " << h_matC[i] - 5.0 << std::endl;
            flag = false;
        }
    }

    if (flag) std::cout << "pass" << std::endl;

    free(h_matA);
    free(h_matB);
    free(h_matC);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    return 0;
}
