#include "wb.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Ядро, выполняется на большом числе нитей 
__global__ void vecAdd(float* in1, float* in2, float* out, int len) {
    // Глобальный индекс нити 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Выполнение обработки соответствующих данной нити данных
    out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* hostInput1;
    float* hostInput2;
    float* hostOutput;
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
        (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
        (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Выделите память GPU
    cudaMalloc((void**)&deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceInput2, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Скопируйте память на GPU
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Инициализируйте размерности сетки и блоков
    dim3 blockDim = dim3(512, 1);
    dim3 gridDim(ceil(((float)inputLength) / ((float)blockDim.x)));

    wbTime_start(Compute, "Performing CUDA computation");
   
    vecAdd << <blockDim, gridDim >> > (deviceInput1, deviceInput2, deviceOutput,
        inputLength);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Скопируйте память GPU обратно на хост
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Освободите память GPU
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}