#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "check.h"


int recursiveReduce(int *data, int const size) {
    // terminate check
    if (size == 1) return data[0];
    int const stride = size / 2;

    if (size % 2 == 1) {
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    }
    else {
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];
        }
    }

    return recursiveReduce(data, stride);
}


__global__ void reduceNeighbored(int * g_idata, int * g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    if (tid >= n) return;
    
    int *idata = g_idata + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}


__global__ void reduceNeigbordLess(int * g_idata, int * g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (tid==0) {
        g_odata[blockIdx.x] = idata[0];
    }
}


__global__ void RedcueInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockDim.x * blockIdx.x;

    if (idx > n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) 
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnroll2(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    if (tid >= n) return;

    int *idata = g_idata + blockDim.x * blockIdx.x * 2;

    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnroll8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    if (tid >= n) return;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrollWrap8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    if (tid >= n) return;

    if (idx + blockDim.x * 7 < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceCompleteUnroll8Wrap(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (tid >= n) return;

    if (idx + 7 * blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        // volatile 修饰的变量，每次都会去内存中拿变量，不会使用寄存器中的值, 并且是顺序执行
        volatile int * vmem = idata;  // 不优化
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];

        // idata[tid] += idata[tid + 32];
        // __syncthreads();
        // idata[tid] += idata[tid + 16];
        // __syncthreads();
        // idata[tid] += idata[tid + 8];
        // __syncthreads();
        // idata[tid] += idata[tid + 4];
        // __syncthreads();
        // idata[tid] += idata[tid + 2];
        // __syncthreads();
        // idata[tid] += idata[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


int main() {
    cudaSetDevice(0);
    
    int size = 1 << 24;
    std::cout << "array size: " << size << std::endl;
    
    dim3 block(1024, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    std::cout << "block size: " << block.x << std::endl;
    std::cout << "grid size: " << grid.x << std::endl;

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    for (int i = 0; i < size; ++i) {
        h_idata[i] = 1;
    }

    int *d_idata, *d_odata;

    CHECK(cudaMalloc((void **)&d_idata, bytes));
    CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    // reduceNeighbored
    // reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);

    // reduceNeighboredLess
    // reduceNeigbordLess<<<grid, block>>>(d_idata, d_odata, size);

    // reduceUnroll
    // reduceUnroll2<<<grid.x / 2, block>>>(d_idata, d_odata, size);

    // reduceUnroll8
    // reduceUnroll8<<<grid.x / 8, block>>>(d_idata, d_odata, size);

    // reduceUnrollWrap8
    // reduceUnrollWrap8<<<grid.x / 8, block>>>(d_idata, d_odata, size);

    // reduceCompleteUnrollWrap8
    reduceCompleteUnroll8Wrap<<<grid.x / 8, block>>>(d_idata, d_odata, size);


    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    std::cout << "gpu sum: " << gpu_sum << std::endl;

    cudaFree(d_idata);
    cudaFree(d_odata);

    free(h_idata);
    free(h_odata);

    return 0;

}