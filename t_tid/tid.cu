#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "check.h"


// Thread 1D
__global__ void thread_1d(const int *d_a, const int *d_b, int *d_c) {
    int tid = threadIdx.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// Thread 2D
__global__ void thread_2d(const int *d_a, const int *d_b, int *d_c) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// Thread 3D
__global__ void thread_3d(const int *d_a, const int *d_b, int *d_c) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block 1D
__global__ void block_1d(const int *d_a, const int *d_b, int *d_c) {
    int tid = blockIdx.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block 2D
__global__ void block_2d(const int *d_a, const int *d_b, int *d_c) {
    int tid = blockIdx.x + blockIdx.y * gridDim.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block 3D
__global__ void block_3d(const int *d_a, const int *d_b, int *d_c) {
    int tid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 1D-1D
__global__ void block_1d_thread_1d(const int *d_a, const int *d_b, int *d_c) {
    int threadId_1D = threadIdx.x;
    int tid = blockDim.x * blockIdx.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 1D-2D
__global__ void block_1d_thread_2d(const int *d_a, const int *d_b, int *d_c) {
    int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadId_2D + (blockDim.x * blockDim.y) * blockIdx.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 1D-3D
__global__ void block_1d_thread_3d(const int *d_a, const int *d_b, int *d_c) {
    int threadId_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int tid = threadId_3D + (blockDim.x * blockDim.y * blockDim.z) * blockIdx.x;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 2D-1D
__global__ void block_2d_thread_1d(const int *d_a, const int *d_b, int *d_c) {
    int blockID_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadIdx.x + blockDim.x * blockID_2D;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 3D-1D
__global__ void block_3d_thread_1d(const int *d_a, const int *d_b, int *d_c) {
    int blockID_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int tid = threadIdx.x + blockDim.x * blockID_3D;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 2D-2D
__global__ void block_2d_thread_2d(const int *d_a, const int *d_b, int *d_c) {
    int threadID_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int blockID_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadID_2D + blockDim.x * blockDim.y * blockID_2D;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 2D-3D
__global__ void block_2d_thread_3d(const int *d_a, const int *d_b, int *d_c) {
    int threadID_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    int blockID_2D = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadID_3D + blockDim.x * blockDim.y * blockDim.z * blockID_2D;
    d_c[tid] = d_a[tid] + d_b[tid];
}


// block-thread 3D-2D
__global__ void block_3d_thread_2d(const int *d_a, const int *d_b, int *d_c) {
    int threadID_2D = threadIdx.x + threadIdx.y * blockDim.x;
    int blockID_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
    int tid = threadID_2D + blockDim.x * blockDim.y * blockID_3D;
    d_c[tid] = d_a[tid] + d_b[tid];
}

// block-thread 3D-3D
__global__ void block_3d_thread_3d(const int *d_a, const int *d_b, int *d_c) {
    int threadID_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    int blockID_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
    int tid = threadID_3D + (blockDim.x * blockDim.y * blockDim.z) * blockID_3D;
    d_c[tid] = d_a[tid] + d_b[tid];
}


void addWithCuda(int *h_c, const int *h_a, const int *h_b, const int size, const int cuda_id = 0) {
    int *d_a, *d_b, *d_c;
    int nByte = size * sizeof(int);

    cudaSetDevice(cuda_id);
    CHECK(cudaMalloc((int **)&d_a, nByte));
    CHECK(cudaMalloc((int **)&d_b, nByte));
    CHECK(cudaMalloc((int **)&d_c, nByte));

    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, nByte, cudaMemcpyHostToDevice));

    // // Thread 1D
    // dim3 block(size);
    // dim3 grid(1);
    // thread_1d<<<grid, block>>>(d_a, d_b, d_c);

    // // Thread 2D
    // dim3 block(size / 20, 20);
    // dim3 grid(1);
    // thread_2d<<<grid, block>>>(d_a, d_b, d_c);

    // // Thread 3D
    // dim3 block(size / 20, 10, 2);
    // dim3 grid(1);
    // thread_3d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 1D
    // dim3 block(1);
    // dim3 grid(size);
    // block_1d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 2D
    // dim3 block(1);
    // dim3 grid(size / 20, 20);
    // block_2d<<<grid, block>>>(d_a, d_b, d_c);

    // //Block 3D
    // dim3 block(1);
    // dim3 grid(size / 20, 10, 2);
    // block_3d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 1D -- Thread 1D
    // dim3 block(size / 30);
    // dim3 grid(30+1);
    // block_1d_thread_1d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 1D -- Thread 2D
    // dim3 block(size / 20, 10);
    // dim3 grid(2);
    // block_1d_thread_2d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 1D -- Thread 3D
    // dim3 block(size / 20, 5, 2);
    // dim3 grid(2);
    // block_1d_thread_3d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 2D -- Thread 1D
    // dim3 block(size / 20);
    // dim3 grid(10, 2);
    // block_2d_thread_1d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 3D -- Thread 1D
    // dim3 block(size / 20);
    // dim3 grid(5, 2, 2);
    // block_3d_thread_1d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 2D -- Thread 2D
    // dim3 block(size / 100, 2);
    // dim3 grid(5, 10);
    // block_2d_thread_2d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 2D -- Thread 3D
    // dim3 block(size / 100, 5, 2);
    // dim3 grid(5, 2);
    // block_2d_thread_3d<<<grid, block>>>(d_a, d_b, d_c);

    // // Block 3D -- Thread 2D
    // dim3 block(size / 100, 5);
    // dim3 grid(2, 2, 5);
    // block_3d_thread_2d<<<grid, block>>>(d_a, d_b, d_c);

    // Block 3D -- Thread 3D
    dim3 block(size / 200, 2, 2);
    dim3 grid(5, 5, 2);
    block_3d_thread_3d<<<grid, block>>>(d_a, d_b, d_c);


    CHECK(cudaMemcpy(h_c, d_c, nByte, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}


int main() {
    int size = 1000;
    int *h_a = (int *)malloc(size * sizeof(int));
    int *h_b = (int *)malloc(size * sizeof(int));
    int *h_c = (int *)malloc(size * sizeof(int));
    int *standard_res = (int *)malloc(size * sizeof(int));
    
    // init matrix
    for (int i = 0; i < size; ++i) {
        h_a[i] = 2;
        h_b[i] = 3;
        standard_res[i] = h_a[i] + h_b[i];
    }

    addWithCuda(h_c, h_a, h_b, 1000);
    CHECK(cudaDeviceReset());

    // // print result
    // for (int i = 0; i < 20; ++i) {
    //     for(int j = 0; j < 50; ++j) {
    //         std::cout << h_c[j + 20 * i] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    bool flag = true;
    for (int i = 0; i < size; ++i) {
        if (standard_res[i] != h_c[i]) {
            flag = false;
            std::cout << "standard res: " << standard_res[i] << " ";
            std::cout << "host c: " << h_c[i] << "  ";
        }
    }
    std::cout << std::endl;

    if (flag)
        std::cout << "pass" << std::endl;
    else
        std::cout << "no pass" << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}



