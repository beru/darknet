#include "avgpool_layer.h"
#include "cuda.h"
#include "xplat.h"
#include <stdio.h>

void make_avgpool_layer(avgpool_layer *l, int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %5d x%5d x%5d   ->  %4d\n",
            w, h, c, c);
    l->type = AVGPOOL;
    l->batch = batch;
    l->h = h;
    l->w = w;
    l->c = c;
    l->out_w = 1;
    l->out_h = 1;
    l->out_c = c;
    l->outputs = l->out_c;
    l->inputs = h * w * c;
    int output_size = l->outputs * batch;
    l->output =  xplat_malloc(output_size, sizeof(float));
    l->delta =   xplat_malloc(output_size, sizeof(float));
    l->forward = forward_avgpool_layer;
    l->backward = backward_avgpool_layer;
#ifdef GPU
    l->forward_gpu = forward_avgpool_layer_gpu;
    l->backward_gpu = backward_avgpool_layer_gpu;
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta, output_size);
#endif
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h * w * l->c;
}

void forward_avgpool_layer(avgpool_layer *l)
{
    network *net = l->net;
    for (int b = 0; b < l->batch; ++b) {
        for (int k = 0; k < l->c; ++k) {
            int out_index = k + b * l->c;
            l->output[out_index] = 0;
            for (int i = 0; i < l->h * l->w; ++i) {
                int in_index = i + l->h * l->w * (k + b * l->c);
                l->output[out_index] += net->input[in_index];
            }
            l->output[out_index] /= l->h * l->w;
        }
    }
}

void backward_avgpool_layer(avgpool_layer *l)
{
    network *net = l->net;
    for (int b = 0; b < l->batch; ++b) {
        for (int k = 0; k < l->c; ++k) {
            int out_index = k + b * l->c;
            for (int i = 0; i < l->h * l->w; ++i) {
                int in_index = i + l->h * l->w * (k + b * l->c);
                net->delta[in_index] += l->delta[out_index] / (l->h * l->w);
            }
        }
    }
}

