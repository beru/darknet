#pragma once

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif

