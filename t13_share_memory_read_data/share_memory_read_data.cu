#include <iostream>
#include "cuda_runtime.h"
#include "check.h"


const int BDIMX = 32;
const int BDIMY = 32;

const int BDIMX_RECT = 32;
const int BDIMY_RECT = 16;
const int IPAD = 1;


__global__ void setRowReadRow(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out) {
    __shared__ int tile[BDIMY][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y + blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadColDyn(int *out) {
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColIpad(int *out) {
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynIpad(int *out) {
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColRect(int *out) {
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColRectDyn(int *out) {
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    unsigned int col_idx = icol * blockDim.x + irow;
    tile[idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColRectPad(int *out) {
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT + IPAD * 2];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColRectDynPad(int *out) {
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    unsigned int row_idx = threadIdx.y * (IPAD + blockDim.x) + threadIdx.x;
    unsigned int col_idx = icol * (IPAD + blockDim.x) + irow;

    tile[row_idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}


int main() {
    cudaSetDevice(0);

    int nElem = BDIMX * BDIMY;
    int nbytes = sizeof(int) * nElem;

    int *out;

    CHECK(cudaMalloc((void **) &out, nbytes));
    
    cudaSharedMemConfig MemConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));

    std::cout << "------------------------" << std::endl;
    switch (MemConfig) {
        case cudaSharedMemBankSizeFourByte:
            std::cout << "the device shared memory is : 4-Byte \n" << std::endl;
            break;
        case cudaSharedMemBankSizeEightByte:
            std::cout << "the device shared memory is : 8-Byte \n" << std::endl;
            break;
    }
    std::cout << "------------------------" << std::endl;

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Row-Row
    // setRowReadRow<<<grid, block>>>(out);

    // Col-Col
    // setColReadCol<<<grid, block>>>(out);

    // Col-Row
    // setColReadRow<<<grid, block>>>(out);

    // Row-Col
    // setRowReadCol<<<grid, block>>>(out);

    // Row-Col-Dyn
    // setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(out);

    // Row-Col-Ipad
    // setRowReadColIpad<<<grid, block>>>(out);

    // Row-Col-Dyn-Ipad
    // setRowReadColDynIpad<<<grid, block, (BDIMX+IPAD) * BDIMY * sizeof(int)>>>(out);

    dim3 block_rect(BDIMX_RECT, BDIMY_RECT);
    dim3 grid_rect(1, 1);
    // Row-Col-Rect
    // setRowReadColRect<<<grid_rect, block_rect>>>(out);

    // Row-Col-Rect-Dyn
    // setRowReadColRectDyn<<<grid_rect, block_rect, BDIMX * BDIMY * sizeof(int)>>>(out);

    // Row-Col-Rect-Pad
    // setRowReadColRectPad<<<grid_rect, block_rect>>>(out);

    // Row-Col-Rect-Dyn-Pad
    setRowReadColRectDynPad<<<grid_rect, block_rect, (BDIMX + 1) * BDIMY * sizeof(int)>>>(out);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elasped_time;
    cudaEventElapsedTime(&elasped_time, start, stop);
    std::cout << "Elasped time: " << elasped_time << "ms" << std::endl;

    cudaFree(out);

    return 0;
}


