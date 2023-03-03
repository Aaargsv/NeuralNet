#ifndef GEMM_H
#define GEMM_H
template <typename T>
void gemm(int M, int N, int K, T *A, int lda, T *B, int ldb, T *C, int ldc);

#endif //GEMM_H
