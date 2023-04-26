#ifndef GEMM_H
#define GEMM_H

#include <vector>

void gemm(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm(int M, int N, int K,
          std::vector<float> &A, int lda,
          std::vector<float> &B, int ldb,
          std::vector<float> &C, int ldc);

#endif //GEMM_H
