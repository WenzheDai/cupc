#include <stdio.h>
#include "cuda_runtime.h"
#include "check.h"


void sumArrays(float *a, float *b, float *res, int offset, const int size)
{

    for(int i=0,k=offset; k<size; ++i, ++k)
    {
        res[i]=a[k]+b[k];
    }

}

__global__ void sumArraysGPU(float *a, float *b, float *res, int offset, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int k = i + offset;

    if(k<n)
        res[i]=a[k]+b[k];
}


int main(int argc,char **argv)
{
    cudaSetDevice(0);

    int nElem=1<<18;
    int offset=0;

    if(argc>=2)
        offset=atoi(argv[1]);
    printf("Vector size:%d\n",nElem);

    int nByte=sizeof(float) * nElem;

    float *a_h=(float*)malloc(nByte);
    float *b_h=(float*)malloc(nByte);
    float *res_h=(float*)malloc(nByte);
    float *res_from_gpu_h=(float*)malloc(nByte);

    memset(res_h,0,nByte);
    memset(res_from_gpu_h,0,nByte);

    for (int i = 0; i < nElem; ++i) {
        a_h[i] = 2.37;
        b_h[i] = 2.33;
    }

    float *a_d,*b_d,*res_d;
    CHECK(cudaMalloc((float**)&a_d,nByte));
    CHECK(cudaMalloc((float**)&b_d,nByte));
    CHECK(cudaMalloc((float**)&res_d,nByte));
    CHECK(cudaMemset(res_d,0,nByte));

    CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem/block.x);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,offset,nElem);
    cudaDeviceSynchronize();
    cudaEventRecord(end);

    CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));

    cudaEventSynchronize(end);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f ms --offset:%d \n",grid.x,block.x,elapsed_time,offset);

    sumArrays(a_h,b_h,res_h,offset,nElem);

    for (int i = 0; i < nElem; ++i) {
        if ((res_h[i] - res_from_gpu_h[i]) > 1e-8) {
            printf("COMPUTE ERROR");
        }
    }

    printf("COMPUTE SUCCESS");

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}