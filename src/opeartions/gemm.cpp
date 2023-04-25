#include "operations/gemm.h"


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

