#include "algorithms/gemm.h"

template <typename T>
void gemm(int M, int N, int K, T *A, int lda, T *B, int ldb, T *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        T *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            c[j] = 0;
        }
        for (int k = 0; k < K; k++ ) {
            T ax = A[i * lda + k];
            const T *b = B + k * ldb;
            for (int j = 0; j < N; j++) {
                c[j] += ax * b[j];
            }
        }
    }
}

