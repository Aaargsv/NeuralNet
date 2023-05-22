#include "operations/gemm.h"
#include <assert.h>
#include <iostream>


void gemm(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            c[j] = 0;
        }
        for (int k = 0; k < K; k++ ) {
            float ax = A[i * lda + k];
            const float *b = B + k * ldb;
            for (int j = 0; j < N; j++) {
                c[j] += ax * b[j];
            }
        }
    }
}


void gemm(int M, int N, int K,
          std::vector<float> &A, int lda,
          std::vector<float> &B, int ldb,
          std::vector<float> &C, int ldc)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = 0;
        }
        for (int k = 0; k < K; k++ ) {

            float ax = A[i * lda + k];
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += ax * B[k * ldb +  j];
            }
        }
    }
}

