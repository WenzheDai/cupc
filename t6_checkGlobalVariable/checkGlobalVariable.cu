#include <cuda_runtime.h>
#include <stdio.h>


__device__ float devData;


__global__ void checkGlobalVaribale() {
    printf("Device: The value of the global variable is %f\n",devData);
    devData+=2.0;
}


int main() {
    float value = 3.14f;

    // 静态分配gpu
    cudaMemcpyToSymbol(devData, &value, sizeof(float));

    printf("Host: copy %f ot the global variable\n", value);

    checkGlobalVaribale<<<1, 1>>>();
    
    // 从静态分配的gpu内存上取数
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));

    printf("Host: the value changed by the kernel to %f \n",value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}