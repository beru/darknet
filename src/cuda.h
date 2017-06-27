#pragma once

#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
float *cuda_make_array(const float *x, size_t n);
int *cuda_make_int_array(const int *x, size_t n);
void cuda_push_array(float *x_gpu, const float *x, size_t n);
void cuda_pull_array(const float *x_gpu, float *x, size_t n);
void cuda_set_device(int n);
void cuda_free(float *x_gpu);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
float cuda_mag_array(float *x_gpu, size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif // #ifdef GPU
