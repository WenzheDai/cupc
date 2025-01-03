#include <cuda_runtime.h>
#include <stdio.h>
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

__global__ void sumArraysGPU(float*a,float*b,float*res,int N)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < N)
        res[i]=a[i]+b[i];
}

int main(int argc,char **argv)
{
    // set up device
    cudaSetDevice(0);

    int nElem=1<<24;
    printf("Vector size:%d\n",nElem);
    int nByte=sizeof(float)*nElem;
    float *res_h=(float*)malloc(nByte);
    memset(res_h, 0, nByte);
    //memset(res_from_gpu_h,0,nByte);

    float *a_d,*b_d,*res_d;

    /* 统一管理 cpu 和 gpu 内存，函数集成了 malloc 和 cudaMemcpy */
    CHECK(cudaMallocManaged((float**)&a_d, nByte));
    CHECK(cudaMallocManaged((float**)&b_d, nByte));
    CHECK(cudaMallocManaged((float**)&res_d, nByte));

    for (int i = 0; i < nElem; ++i) {
        a_d[i] = 2.37;
        b_d[i] = 2.23;
    }

    //CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

    dim3 block(512);
    dim3 grid((nElem-1)/block.x+1);

    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize();

    //CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
    sumArrays(a_d, b_d, res_h,nElem);

    bool flag = true;
    for (int i = 0; i < nElem; ++i) {
        if (abs(res_h[i] - res_d[i]) > 1e-8) {
            printf("ERROR");
            flag = false;
            break;
        }

    }

    if (flag) printf("SUCCESS");

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(res_h);

    return 0;
}