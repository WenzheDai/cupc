#include <iostream>
#include <random>
#include "check.h"


void transformMat2DCPU(float *MatA, float *MatB, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            MatB[i * ny + j] = MatA[j * nx + i];
        }
    }
}

__global__ void copyRow(float *MatA, float *MatB, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * nx;
    if (ix < nx && iy < ny)
        MatB[idx] = MatA[idx];
}

__global__ void copyCol(float *MatA, float *MatB, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix * ny + iy;
    if (ix < nx && iy < ny)
        MatB[idx] = MatA[idx];
}

__global__ void transformNativeRow(float *MatA, float *MatB, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx_row = ix + iy * nx;   // MatA 中idx
    int idx_col = iy + ix * ny;   // 转置后 MatB中 idx

    if (ix < nx && iy < ny) {
        MatB[idx_col] = MatA[idx_row];
    }
}

__global__ void transformNativeRowRoll(float *MatA, float *MatB, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx_row = ix + iy * nx;
    int idx_col = iy + ix * ny;

    if (ix < nx && iy < ny) {
        MatB[idx_col] = MatA[idx_row];
        MatB[idx_col + ny * 1 * blockDim.x] = MatA[idx_row + 1 * blockDim.x];
        MatB[idx_col + ny * 2 * blockDim.x] = MatA[idx_row + 2 * blockDim.x];
        MatB[idx_col + ny * 3 * blockDim.x] = MatA[idx_row + 3 * blockDim.x];
    }
}


int main() {
    // random number
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    cudaSetDevice(0);
    int nx = 1 << 12;
    int ny = 1 << 10;

    int bytes = sizeof(float) * nx * ny;

    float *MatA = (float *)malloc(bytes);
    float *MatB = (float *)malloc(bytes);
    float *from_gpu_Mat = (float *)malloc(bytes);

    for (int i = 0; i < nx * ny; ++i) {
        MatA[i] = dis(gen);
    }

    transformMat2DCPU(MatA, MatB, nx, ny);

    float *d_MatA, *d_MatB;
    CHECK(cudaMalloc((void**)&d_MatA, bytes));
    CHECK(cudaMalloc((void**)&d_MatB, bytes));

    CHECK(cudaMemcpy(d_MatA, MatA, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

    // Native
    // transformNativeRow<<<grid, block>>>(d_MatA, d_MatB, nx, ny);

    // Roll
    dim3 block_roll(32, 32);
    dim3 gird_roll((nx - 1) / (block.x * 4) + 1, (ny - 1) / block.y + 1);
    transformNativeRowRoll<<<gird_roll, block_roll>>>(d_MatA, d_MatB, nx, ny);
    CHECK(cudaMemcpy(from_gpu_Mat, d_MatB, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    bool flag = true;
    for (int i = 0; i < nx * ny; ++i) {
        if (abs(from_gpu_Mat[i] - MatB[i]) > 1e-8) {
            printf("ERROR  ");
            flag = false;
            break;
        }
    }

    if (flag) std::cout << "RESULT SUCCESS" << std::endl;

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    free(MatA);
    free(MatB);
    free(from_gpu_Mat);

    return 0;
}