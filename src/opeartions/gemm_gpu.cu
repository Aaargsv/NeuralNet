#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu.cuh"

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc)
{

    int init[16] = {0};
    cublasHandle_t handles[16];
    int i = 0;
    cudaGetDevice(&i);
    if(!init[i]) {
        cublasCreate(&handles[i]);
        init[i] = 1;
    }
    cublasHandle_t handle = handles[i];

    cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
}