#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "gpu.cuh"

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc)
{


    cublasHandle_t handle;

    if(get_blas_handle(handle)) {
        std::cout << "[Error]: can't run gemm_gpu" << std::endl;
        exit(EXIT_FAILURE);
    }

    cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);

}