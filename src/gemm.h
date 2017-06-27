#pragma once

void gemm_bin(int M, int N, int K, float ALPHA, 
              const char  *A, int lda, 
              const float *B, int ldb,
              float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
          const float *A, int lda, 
          const float *B, int ldb,
          float BETA,
          float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
              const float *A, int lda, 
              const float *B, int ldb,
              float BETA,
              float *C, int ldc);

#ifdef GPU
void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
                const float *A_gpu, int lda, 
                const float *B_gpu, int ldb,
                float BETA,
                float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
              const float *A, int lda, 
              const float *B, int ldb,
              float BETA,
              float *C, int ldc);
#endif // #ifdef GPU

