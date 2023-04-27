#ifndef GPU_H
#define GPU_H
#include <cstdio>


#define BLOCK 512

void display_header();
int gpu_malloc(float *data, int len);
int copy_to_gpu(float *gpu_data, float *cpu_data, int len);
int extract_from_gpu(float *cpu_data, float *gpu_data, int len);

#endif