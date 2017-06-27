#include "rnn_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include "xplat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs * l->batch * steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}

void make_rnn_layer(layer *l,
                    int batch, int inputs, int hidden, int outputs, int steps,
                    ACTIVATION activation, int batch_normalize, int log)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n",
            inputs, outputs);
    batch = batch / steps;
    l->batch = batch;
    l->type = RNN;
    l->steps = steps;
    l->hidden = hidden;
    l->inputs = inputs;

    l->state = xplat_malloc(batch * hidden * (steps + 1), sizeof(float));

    l->input_layer = xplat_malloc(1, sizeof(layer));
    fprintf(stderr, "\t\t");
    make_connected_layer(l->input_layer, batch * steps, inputs, hidden, activation, batch_normalize);
    l->input_layer->batch = batch;

    l->self_layer = xplat_malloc(1, sizeof(layer));
    fprintf(stderr, "\t\t");
    make_connected_layer(l->self_layer, batch * steps, hidden, hidden, (log == 2) ? LOGGY : (log == 1 ? LOGISTIC : activation), batch_normalize);
    l->self_layer->batch = batch;

    l->output_layer = xplat_malloc(1, sizeof(layer));
    fprintf(stderr, "\t\t");
    make_connected_layer(l->output_layer, batch * steps, hidden, outputs, activation, batch_normalize);
    l->output_layer->batch = batch;

    l->outputs = outputs;
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    l->forward = forward_rnn_layer;
    l->backward = backward_rnn_layer;
    l->update = update_rnn_layer;
#ifdef GPU
    l->forward_gpu = forward_rnn_layer_gpu;
    l->backward_gpu = backward_rnn_layer_gpu;
    l->update_gpu = update_rnn_layer_gpu;
    l->state_gpu = cuda_make_array(l->state, batch * hidden * (steps + 1));
    l->output_gpu = l->output_layer->output_gpu;
    l->delta_gpu = l->output_layer->delta_gpu;
#endif
}

void update_rnn_layer(layer *l, int batch, float learning_rate, float momentum, float decay)
{
    update_connected_layer(l->input_layer, batch, learning_rate, momentum, decay);
    update_connected_layer(l->self_layer, batch, learning_rate, momentum, decay);
    update_connected_layer(l->output_layer, batch, learning_rate, momentum, decay);
}

void forward_rnn_layer(layer *l)
{
    network *net = l->net;
    layer *input_layer = l->input_layer;
    layer *self_layer = l->self_layer;
    layer *output_layer = l->output_layer;

    fill_cpu(l->outputs * l->batch * l->steps, 0, output_layer->delta, 1);
    fill_cpu(l->hidden * l->batch * l->steps, 0, self_layer->delta, 1);
    fill_cpu(l->hidden * l->batch * l->steps, 0, input_layer->delta, 1);
    if (net->train) fill_cpu(l->hidden * l->batch, 0, l->state, 1);

    for (int i = 0; i < l->steps; ++i) {
        float *input_org = net->input;
        forward_connected_layer(input_layer);

        net->input = l->state;
        forward_connected_layer(self_layer);

        float *old_state = l->state;
        if (net->train) l->state += l->hidden * l->batch;
        if (l->shortcut) {
            copy_cpu(l->hidden * l->batch, old_state, 1, l->state, 1);
        }else {
            fill_cpu(l->hidden * l->batch, 0, l->state, 1);
        }
        axpy_cpu(l->hidden * l->batch, 1, input_layer->output, 1, l->state, 1);
        axpy_cpu(l->hidden * l->batch, 1, self_layer->output, 1, l->state, 1);

        net->input = l->state;
        forward_connected_layer(output_layer);

        net->input = input_org;
        net->input += l->inputs * l->batch;
        increment_layer(input_layer, 1);
        increment_layer(self_layer, 1);
        increment_layer(output_layer, 1);
    }
}

void backward_rnn_layer(layer *l)
{
    network *net = l->net;
    layer *input_layer = l->input_layer;
    layer *self_layer = l->self_layer;
    layer *output_layer = l->output_layer;

    increment_layer(input_layer, l->steps - 1);
    increment_layer(self_layer, l->steps - 1);
    increment_layer(output_layer, l->steps - 1);

    l->state += l->hidden * l->batch * l->steps;
    for (int i = l->steps - 1; i >= 0; --i) {
        copy_cpu(l->hidden * l->batch, input_layer->output, 1, l->state, 1);
        axpy_cpu(l->hidden * l->batch, 1, self_layer->output, 1, l->state, 1);

        float *input_org = net->input;
        float *delta_org = net->delta;
        net->input = l->state;
        net->delta = self_layer->delta;
        backward_connected_layer(output_layer);

        l->state -= l->hidden * l->batch;
        /*
           if (i > 0) {
           copy_cpu(l->hidden * l->batch, input_layer.output - l->hidden*l->batch, 1, l->state, 1);
           axpy_cpu(l->hidden * l->batch, 1, self_layer.output - l->hidden*l->batch, 1, l->state, 1);
           }else {
           fill_cpu(l->hidden * l->batch, 0, l->state, 1);
           }
         */

        net->input = l->state;
        net->delta = self_layer->delta - l->hidden * l->batch;
        if (i == 0) net->delta = 0;
        backward_connected_layer(self_layer);

        copy_cpu(l->hidden * l->batch, self_layer->delta, 1, input_layer->delta, 1);
        if (i > 0 && l->shortcut) {
            axpy_cpu(l->hidden * l->batch, 1, self_layer->delta, 1, self_layer->delta - l->hidden * l->batch, 1);
        }
        net->input = input_org + i * l->inputs * l->batch;
        if (delta_org) net->delta = delta_org + i * l->inputs * l->batch;
        else net->delta = 0;
        backward_connected_layer(input_layer);

        net->input = input_org;
        net->delta = delta_org;

        increment_layer(input_layer, -1);
        increment_layer(self_layer, -1);
        increment_layer(output_layer, -1);
    }
}

#ifdef GPU

void pull_rnn_layer(layer *l)
{
    pull_connected_layer(l->input_layer);
    pull_connected_layer(l->self_layer);
    pull_connected_layer(l->output_layer);
}

void push_rnn_layer(layer *l)
{
    push_connected_layer(l->input_layer);
    push_connected_layer(l->self_layer);
    push_connected_layer(l->output_layer);
}

void update_rnn_layer_gpu(layer *l, int batch, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(l->input_layer, batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->self_layer, batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->output_layer, batch, learning_rate, momentum, decay);
}

void forward_rnn_layer_gpu(layer *l)
{
    network *net = l->net;
    layer *input_layer = l->input_layer;
    layer *self_layer = l->self_layer;
    layer *output_layer = l->output_layer;

    fill_ongpu(l->outputs * l->batch * l->steps, 0, output_layer->delta_gpu, 1);
    fill_ongpu(l->hidden * l->batch * l->steps, 0, self_layer->delta_gpu, 1);
    fill_ongpu(l->hidden * l->batch * l->steps, 0, input_layer->delta_gpu, 1);
    if (net->train) fill_ongpu(l->hidden * l->batch, 0, l->state_gpu, 1);

    for (int i = 0; i < l->steps; ++i) {
        float *input_gpu_org = net->input_gpu;
        forward_connected_layer_gpu(input_layer);

        net->input_gpu = l->state_gpu;
        forward_connected_layer_gpu(self_layer);

        float *old_state = l->state_gpu;
        if (net->train) l->state_gpu += l->hidden * l->batch;
        if (l->shortcut) {
            copy_ongpu(l->hidden * l->batch, old_state, 1, l->state_gpu, 1);
        }else {
            fill_ongpu(l->hidden * l->batch, 0, l->state_gpu, 1);
        }
        axpy_ongpu(l->hidden * l->batch, 1, input_layer->output_gpu, 1, l->state_gpu, 1);
        axpy_ongpu(l->hidden * l->batch, 1, self_layer->output_gpu, 1, l->state_gpu, 1);

        net->input_gpu = l->state_gpu;
        forward_connected_layer_gpu(output_layer);

        net->input_gpu = input_gpu_org;
        net->input_gpu += l->inputs * l->batch;
        increment_layer(input_layer, 1);
        increment_layer(self_layer, 1);
        increment_layer(output_layer, 1);
    }
}

void backward_rnn_layer_gpu(layer *l)
{
    network *net = l->net;
    layer *input_layer = l->input_layer;
    layer *self_layer = l->self_layer;
    layer *output_layer = l->output_layer;
    increment_layer(input_layer,  l->steps - 1);
    increment_layer(self_layer,   l->steps - 1);
    increment_layer(output_layer, l->steps - 1);
    l->state_gpu += l->hidden * l->batch * l->steps;
    for (int i = l->steps - 1; i >= 0; --i) {
        float *input_gpu_org = net->input_gpu;
        float *delta_gpu_org = net->delta_gpu;

        net->input_gpu = l->state_gpu;
        net->delta_gpu = self_layer->delta_gpu;
        backward_connected_layer_gpu(output_layer);

        l->state_gpu -= l->hidden * l->batch;

        copy_ongpu(l->hidden*l->batch, self_layer->delta_gpu, 1, input_layer->delta_gpu, 1);

        net->input_gpu = l->state_gpu;
        net->delta_gpu = self_layer->delta_gpu - l->hidden * l->batch;
        if (i == 0) net->delta_gpu = 0;
        backward_connected_layer_gpu(self_layer);

        //copy_ongpu(l->hidden * l->batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
        if (i > 0 && l->shortcut) {
            axpy_ongpu(l->hidden * l->batch, 1, self_layer->delta_gpu,
                       1, self_layer->delta_gpu - l->hidden * l->batch, 1);
        }
        net->input_gpu = input_gpu_org + i * l->inputs * l->batch;
        if (delta_gpu_org) net->delta_gpu = delta_gpu_org + i * l->inputs * l->batch;
        else net->delta_gpu = 0;
        backward_connected_layer_gpu(input_layer);

        net->input_gpu = input_gpu_org;
        net->delta_gpu = delta_gpu_org;
        increment_layer(input_layer,  -1);
        increment_layer(self_layer,   -1);
        increment_layer(output_layer, -1);
    }
}

#endif  // #ifdef GPU
