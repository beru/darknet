#include "normalization_layer.h"
#include "blas.h"

#include <stdio.h>

void make_normalization_layer(layer *l,
                              int batch, int w, int h, int c,
                              int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n",
            w, h, c, size);
    l->type = NORMALIZATION;
    l->batch = batch;
    l->h = l->out_h = h;
    l->w = l->out_w = w;
    l->c = l->out_c = c;
    l->kappa = kappa;
    l->size = size;
    l->alpha = alpha;
    l->beta = beta;
    l->output = xplat_malloc(h * w * c * batch, sizeof(float));
    l->delta = xplat_malloc(h * w * c * batch, sizeof(float));
    l->squared = xplat_malloc(h * w * c * batch, sizeof(float));
    l->norms = xplat_malloc(h * w * c * batch, sizeof(float));
    l->inputs = w * h * c;
    l->outputs = l->inputs;

    l->forward = forward_normalization_layer;
    l->backward = backward_normalization_layer;
#ifdef GPU
    l->forward_gpu = forward_normalization_layer_gpu;
    l->backward_gpu = backward_normalization_layer_gpu;

    l->output_gpu =  cuda_make_array(l->output, h * w * c * batch);
    l->delta_gpu =   cuda_make_array(l->delta, h * w * c * batch);
    l->squared_gpu = cuda_make_array(l->squared, h * w * c * batch);
    l->norms_gpu =   cuda_make_array(l->norms, h * w * c * batch);
#endif  // #ifdef GPU
}

void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w * h * c;
    layer->outputs = layer->inputs;
    layer->output = realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = realloc(layer->norms, h * w * c * batch * sizeof(float));
#ifdef GPU
    cuda_free(layer->output_gpu);
    cuda_free(layer->delta_gpu); 
    cuda_free(layer->squared_gpu); 
    cuda_free(layer->norms_gpu);   
    layer->output_gpu =  cuda_make_array(layer->output, h * w * c * batch);
    layer->delta_gpu =   cuda_make_array(layer->delta, h * w * c * batch);
    layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
    layer->norms_gpu =   cuda_make_array(layer->norms, h * w * c * batch);
#endif  // #ifdef GPU
}

void forward_normalization_layer(layer *l, network *net)
{
    int k, b;
    int w = l->w;
    int h = l->h;
    int c = l->c;
    scal_cpu(w * h * c * l->batch, 0, l->squared, 1);

    for (b = 0; b < l->batch; ++b) {
        float *squared = l->squared + w * h * c * b;
        float *norms   = l->norms + w * h * c * b;
        float *input   = net->input + w * h * c * b;
        pow_cpu(w * h * c, 2, input, 1, squared, 1);

        const_cpu(w * h, l->kappa, norms, 1);
        for (k = 0; k < l->size / 2; ++k) {
            axpy_cpu(w*h, l->alpha, squared + w*h*k, 1, norms, 1);
        }

        for (k = 1; k < l->c; ++k) {
            copy_cpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
            int prev = k - ((l->size - 1) / 2) - 1;
            int next = k + (l->size / 2);
            if (prev >= 0)      axpy_cpu(w * h, -l->alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
            if (next < l->c) axpy_cpu(w * h,  l->alpha, squared + w * h * next, 1, norms + w * h * k, 1);
        }
    }
    pow_cpu(w*h*c*l->batch, -l->beta, l->norms, 1, l->output, 1);
    mul_cpu(w*h*c*l->batch, net->input, 1, l->output, 1);
}

void backward_normalization_layer(layer *l, network *net)
{
    // TODO This is approximate ;-)
    // Also this should add in to delta instead of overwritting.

    int w = l->w;
    int h = l->h;
    int c = l->c;
    pow_cpu(w * h * c * l->batch, -l->beta, l->norms, 1, net->delta, 1);
    mul_cpu(w * h * c * l->batch, l->delta, 1, net->delta, 1);
}

#ifdef GPU
void forward_normalization_layer_gpu(layer *l, network *net)
{
    int k, b;
    int w = l->w;
    int h = l->h;
    int c = l->c;
    scal_ongpu(w * h * c * l->batch, 0, l->squared_gpu, 1);

    for (b = 0; b < l->batch; ++b) {
        float *squared = l->squared_gpu + w * h * c * b;
        float *norms   = l->norms_gpu + w * h * c * b;
        float *input   = net->input_gpu + w * h * c * b;
        pow_ongpu(w * h * c, 2, input, 1, squared, 1);

        const_ongpu(w * h, l->kappa, norms, 1);
        for (k = 0; k < l->size / 2; ++k) {
            axpy_ongpu(w * h, l->alpha, squared + w * h * k, 1, norms, 1);
        }

        for (k = 1; k < l->c; ++k) {
            copy_ongpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
            int prev = k - ((l->size - 1) / 2) - 1;
            int next = k + (l->size / 2);
            if (prev >= 0)      axpy_ongpu(w * h, -l->alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
            if (next < l->c) axpy_ongpu(w * h,  l->alpha, squared + w * h * next, 1, norms + w * h * k, 1);
        }
    }
    pow_ongpu(w * h * c * l->batch, -l->beta, l->norms_gpu, 1, l->output_gpu, 1);
    mul_ongpu(w * h * c * l->batch, net->input_gpu, 1, l->output_gpu, 1);
}

void backward_normalization_layer_gpu(layer *l, network *net)
{
    // TODO This is approximate ;-)

    int w = l->w;
    int h = l->h;
    int c = l->c;
    pow_ongpu(w * h * c * l->batch, -l->beta, l->norms_gpu, 1, net->delta_gpu, 1);
    mul_ongpu(w * h * c * l->batch, l->delta_gpu, 1, net->delta_gpu, 1);
}

#endif  // #ifdef GPU
