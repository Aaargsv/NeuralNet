#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include "gpu.cuh"

void display_header()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;
    //std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl << std::endl;

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << std::endl;
    }
}

int cuda_get_device(int &n)
{

    cudaError_t error_status = cudaGetDevice(&n);
    std::cout << "n = " << n << std::endl;
    if (error_status != cudaSuccess) {
        std::cout << "[Error]: cuda can't get device. [Cuda error]: "
                  << cudaGetErrorString( error_status ) << std::endl;
        return 1;
    }
    return 0;
}

int get_blas_handle(cublasHandle_t &hd)
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int n = 0;
    if (cuda_get_device(n)) {
        std::cout << "[Error]: can't get_blas_handle." << std::endl;
        return 1;
    }

    std::cout << "get_blas_handle n = " << n << std::endl;

    if(!init[n]) {
        cublasStatus_t error_status = cublasCreate(&handle[n]);
        if (error_status !=  CUBLAS_STATUS_SUCCESS) {
            std::cout << "[Error]: CUBLAS initialization failed." << std::endl;
            return 1;
        }
        init[n] = 1;
    }
    hd = handle[n];
    return 0;
}

int gpu_malloc(float **data, int len)
{
    cudaError_t error_status = cudaMalloc((void **)data, len);
    if (error_status != cudaSuccess) {
        std::cout << "[Error]: cuda can't malloc. [Cuda error]: "
                  << cudaGetErrorString( error_status ) << std::endl;
        return 1;
    }
    return 0;
}

int copy_to_gpu(float *gpu_data, float *cpu_data, int len)
{
    cudaError_t error_status = cudaMemcpy(gpu_data, cpu_data, len, cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        std::cout << "[Error]: cuda can't copy to gpu. [Cuda error]: "
                  << cudaGetErrorString( error_status ) << std::endl;
        return 1;
    }
    return 0;
}

int extract_from_gpu(float *cpu_data, float *gpu_data, int len)
{
    cudaError_t error_status = cudaMemcpy(cpu_data , gpu_data, len, cudaMemcpyDeviceToHost);
    if (error_status != cudaSuccess) {
        std::cout << "[Error]: cuda can't extract from gpu. [Cuda error]: "
                  << cudaGetErrorString( error_status ) << std::endl;
        return 1;
    }
    return 0;
}



