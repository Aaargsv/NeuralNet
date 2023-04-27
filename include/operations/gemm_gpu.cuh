#ifndef GEMM_GPU_H
#define GEMM_GPU_H

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc);

#endif //GEMM_GPU_H
