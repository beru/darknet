#include "connected_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void make_connected_layer(connected_layer *l,
                          int batch, int inputs, int outputs,
                          ACTIVATION activation, int batch_normalize)
{
    l->type = CONNECTED;

    l->inputs = inputs;
    l->outputs = outputs;
    l->batch = batch;
    l->batch_normalize = batch_normalize;
    l->h = 1;
    l->w = 1;
    l->c = inputs;
    l->out_h = 1;
    l->out_w = 1;
    l->out_c = outputs;

    l->output = xplat_malloc(batch * outputs, sizeof(float));
    l->delta = xplat_malloc(batch * outputs, sizeof(float));

    l->weight_updates = xplat_malloc(inputs * outputs, sizeof(float));
    l->bias_updates = xplat_malloc(outputs, sizeof(float));

    l->weights = xplat_malloc(outputs * inputs, sizeof(float));
    l->biases = xplat_malloc(outputs, sizeof(float));

    l->forward = forward_connected_layer;
    l->backward = backward_connected_layer;
    l->update = update_connected_layer;

    //float scale = 1. / sqrt(inputs);
    float scale = sqrt(2. / inputs);
    for (int i = 0; i < outputs * inputs; ++i) {
        l->weights[i] = scale * rand_uniform(-1, 1);
    }

    for (int i = 0; i < outputs; ++i) {
        l->biases[i] = 0;
    }

    if (batch_normalize) {
        l->scales = xplat_malloc(outputs, sizeof(float));
        l->scale_updates = xplat_malloc(outputs, sizeof(float));
        for (int i = 0; i < outputs; ++i) {
            l->scales[i] = 1;
        }

        l->mean = xplat_malloc(outputs, sizeof(float));
        l->mean_delta = xplat_malloc(outputs, sizeof(float));
        l->variance = xplat_malloc(outputs, sizeof(float));
        l->variance_delta = xplat_malloc(outputs, sizeof(float));

        l->rolling_mean = xplat_malloc(outputs, sizeof(float));
        l->rolling_variance = xplat_malloc(outputs, sizeof(float));

        l->x = xplat_malloc(batch * outputs, sizeof(float));
        l->x_norm = xplat_malloc(batch * outputs, sizeof(float));
    }

#ifdef GPU
    l->forward_gpu = forward_connected_layer_gpu;
    l->backward_gpu = backward_connected_layer_gpu;
    l->update_gpu = update_connected_layer_gpu;

    l->weights_gpu = cuda_make_array(l->weights, outputs * inputs);
    l->biases_gpu = cuda_make_array(l->biases, outputs);

    l->weight_updates_gpu = cuda_make_array(l->weight_updates, outputs * inputs);
    l->bias_updates_gpu = cuda_make_array(l->bias_updates, outputs);

    l->output_gpu = cuda_make_array(l->output, outputs * batch);
    l->delta_gpu = cuda_make_array(l->delta, outputs * batch);
    if (batch_normalize) {
        l->scales_gpu = cuda_make_array(l->scales, outputs);
        l->scale_updates_gpu = cuda_make_array(l->scale_updates, outputs);

        l->mean_gpu = cuda_make_array(l->mean, outputs);
        l->variance_gpu = cuda_make_array(l->variance, outputs);

        l->rolling_mean_gpu = cuda_make_array(l->mean, outputs);
        l->rolling_variance_gpu = cuda_make_array(l->variance, outputs);

        l->mean_delta_gpu = cuda_make_array(l->mean, outputs);
        l->variance_delta_gpu = cuda_make_array(l->variance, outputs);

        l->x_gpu = cuda_make_array(l->output, l->batch * outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch * outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l->normTensorDesc);
        cudnnCreateTensorDescriptor(&l->dstTensorDesc);
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, l->out_c, 1, 1); 
#endif
}
#endif  // #ifdef GPU
    l->activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n",
            inputs, outputs);
}

void update_connected_layer(connected_layer *l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_cpu(l->outputs, learning_rate / batch, l->bias_updates, 1, l->biases, 1);
    scal_cpu(l->outputs, momentum, l->bias_updates, 1);

    if (l->batch_normalize) {
        axpy_cpu(l->outputs, learning_rate / batch, l->scale_updates, 1, l->scales, 1);
        scal_cpu(l->outputs, momentum, l->scale_updates, 1);
    }

    axpy_cpu(l->inputs * l->outputs, -decay * batch, l->weights, 1, l->weight_updates, 1);
    axpy_cpu(l->inputs * l->outputs, learning_rate / batch, l->weight_updates, 1, l->weights, 1);
    scal_cpu(l->inputs * l->outputs, momentum, l->weight_updates, 1);
}

void forward_connected_layer(connected_layer *l, network *net)
{
    fill_cpu(l->outputs * l->batch, 0, l->output, 1);
    int m = l->batch;
    int k = l->inputs;
    int n = l->outputs;
    float *a = net->input;
    float *b = l->weights;
    float *c = l->output;
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    if (l->batch_normalize) {
        if (net->train) {
            mean_cpu(l->output, l->batch, l->outputs, 1, l->mean);
            variance_cpu(l->output, l->mean, l->batch, l->outputs, 1, l->variance);

            scal_cpu(l->outputs, .95, l->rolling_mean, 1);
            axpy_cpu(l->outputs, .05, l->mean, 1, l->rolling_mean, 1);
            scal_cpu(l->outputs, .95, l->rolling_variance, 1);
            axpy_cpu(l->outputs, .05, l->variance, 1, l->rolling_variance, 1);

            copy_cpu(l->outputs*l->batch, l->output, 1, l->x, 1);
            normalize_cpu(l->output, l->mean, l->variance, l->batch, l->outputs, 1);   
            copy_cpu(l->outputs*l->batch, l->output, 1, l->x_norm, 1);
        }else {
            normalize_cpu(l->output, l->rolling_mean, l->rolling_variance, l->batch, l->outputs, 1);
        }
        scale_bias(l->output, l->scales, l->batch, l->outputs, 1);
    }
    for (int i = 0; i < l->batch; ++i) {
        axpy_cpu(l->outputs, 1, l->biases, 1, l->output + i * l->outputs, 1);
    }
    activate_array(l->output, l->outputs * l->batch, l->activation);
}

void backward_connected_layer(connected_layer *l, network *net)
{
    gradient_array(l->output, l->outputs * l->batch, l->activation, l->delta);
    for (int i = 0; i < l->batch; ++i) {
        axpy_cpu(l->outputs, 1, l->delta + i * l->outputs, 1, l->bias_updates, 1);
    }
    if (l->batch_normalize) {
        backward_scale_cpu(l->x_norm, l->delta, l->batch, l->outputs, 1, l->scale_updates);

        scale_bias(l->delta, l->scales, l->batch, l->outputs, 1);

        mean_delta_cpu(l->delta, l->variance, l->batch, l->outputs, 1, l->mean_delta);
        variance_delta_cpu(l->x, l->delta, l->mean, l->variance, l->batch, l->outputs, 1, l->variance_delta);
        normalize_delta_cpu(l->x, l->mean, l->variance, l->mean_delta, l->variance_delta, l->batch, l->outputs, 1, l->delta);
    }

    int m = l->outputs;
    int k = l->batch;
    int n = l->inputs;
    float *a = l->delta;
    float *b = net->input;
    float *c = l->weight_updates;
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l->batch;
    k = l->outputs;
    n = l->inputs;

    a = l->delta;
    b = l->weights;
    c = net->delta;

    if (c) {
        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
}


void denormalize_connected_layer(layer *l)
{
    for (int i = 0; i < l->outputs; ++i) {
        float scale = l->scales[i] / sqrt(l->rolling_variance[i] + .000001);
        for (int j = 0; j < l->inputs; ++j) {
            l->weights[i * l->inputs + j] *= scale;
        }
        l->biases[i] -= l->rolling_mean[i] * scale;
        l->scales[i] = 1;
        l->rolling_mean[i] = 0;
        l->rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer *l)
{
    if (l->batch_normalize) {
        printf("Scales ");
        print_statistics(l->scales, l->outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l->rolling_mean, l->outputs);
           printf("Rolling Variance ");
           print_statistics(l->rolling_variance, l->outputs);
         */
    }
    printf("Biases ");
    print_statistics(l->biases, l->outputs);
    printf("Weights ");
    print_statistics(l->weights, l->outputs);
}

#ifdef GPU

void pull_connected_layer(connected_layer *l)
{
    cuda_pull_array(l->weights_gpu, l->weights, l->inputs * l->outputs);
    cuda_pull_array(l->biases_gpu, l->biases, l->outputs);
    cuda_pull_array(l->weight_updates_gpu, l->weight_updates, l->inputs * l->outputs);
    cuda_pull_array(l->bias_updates_gpu, l->bias_updates, l->outputs);
    if (l->batch_normalize) {
        cuda_pull_array(l->scales_gpu, l->scales, l->outputs);
        cuda_pull_array(l->rolling_mean_gpu, l->rolling_mean, l->outputs);
        cuda_pull_array(l->rolling_variance_gpu, l->rolling_variance, l->outputs);
    }
}

void push_connected_layer(connected_layer *l)
{
    cuda_push_array(l->weights_gpu, l->weights, l->inputs * l->outputs);
    cuda_push_array(l->biases_gpu, l->biases, l->outputs);
    cuda_push_array(l->weight_updates_gpu, l->weight_updates, l->inputs * l->outputs);
    cuda_push_array(l->bias_updates_gpu, l->bias_updates, l->outputs);
    if (l->batch_normalize) {
        cuda_push_array(l->scales_gpu, l->scales, l->outputs);
        cuda_push_array(l->rolling_mean_gpu, l->rolling_mean, l->outputs);
        cuda_push_array(l->rolling_variance_gpu, l->rolling_variance, l->outputs);
    }
}

void update_connected_layer_gpu(connected_layer *l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_ongpu(l->outputs, learning_rate / batch, l->bias_updates_gpu, 1, l->biases_gpu, 1);
    scal_ongpu(l->outputs, momentum, l->bias_updates_gpu, 1);

    if (l->batch_normalize) {
        axpy_ongpu(l->outputs, learning_rate / batch, l->scale_updates_gpu, 1, l->scales_gpu, 1);
        scal_ongpu(l->outputs, momentum, l->scale_updates_gpu, 1);
    }

    axpy_ongpu(l->inputs * l->outputs, -decay * batch, l->weights_gpu, 1, l->weight_updates_gpu, 1);
    axpy_ongpu(l->inputs * l->outputs, learning_rate / batch, l->weight_updates_gpu, 1, l->weights_gpu, 1);
    scal_ongpu(l->inputs * l->outputs, momentum, l->weight_updates_gpu, 1);
}

void forward_connected_layer_gpu(connected_layer *l, network *net)
{
    fill_ongpu(l->outputs * l->batch, 0, l->output_gpu, 1);

    int m = l->batch;
    int k = l->inputs;
    int n = l->outputs;
    float * a = net->input_gpu;
    float * b = l->weights_gpu;
    float * c = l->output_gpu;
    gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    if (l->batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    }
    for (int i = 0; i < l->batch; ++i) {
        axpy_ongpu(l->outputs, 1, l->biases_gpu, 1, l->output_gpu + i * l->outputs, 1);
    }
    activate_array_ongpu(l->output_gpu, l->outputs * l->batch, l->activation);
}

void backward_connected_layer_gpu(connected_layer *l, network *net)
{
    constrain_ongpu(l->outputs * l->batch, 1, l->delta_gpu, 1);
    gradient_array_ongpu(l->output_gpu, l->outputs * l->batch, l->activation, l->delta_gpu);
    for (int i = 0; i < l->batch; ++i) {
        axpy_ongpu(l->outputs, 1, l->delta_gpu + i * l->outputs, 1, l->bias_updates_gpu, 1);
    }

    if (l->batch_normalize) {
        backward_batchnorm_layer_gpu(l, net);
    }

    int m = l->outputs;
    int k = l->batch;
    int n = l->inputs;
    float * a = l->delta_gpu;
    float * b = net->input_gpu;
    float * c = l->weight_updates_gpu;
    gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l->batch;
    k = l->outputs;
    n = l->inputs;

    a = l->delta_gpu;
    b = l->weights_gpu;
    c = net->delta_gpu;

    if (c) {
        gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
}

#endif // #ifdef GPU
