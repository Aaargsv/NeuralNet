#ifndef GPU_H
#define GPU_H
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define BLOCK 512

void display_header();
int gpu_malloc(float **data, int len);
int copy_to_gpu(float *gpu_data, float *cpu_data, int len);
int extract_from_gpu(float *cpu_data, float *gpu_data, int len);
int get_blas_handle(cublasHandle_t &hd);
int cuda_get_device(int &n);
int gpu_free_memory(float *dev_data);

#endif