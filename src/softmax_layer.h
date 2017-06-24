#pragma once

#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
void make_softmax_layer(softmax_layer *l, int batch, int inputs, int groups);
void forward_softmax_layer(softmax_layer *l, network *net);
void backward_softmax_layer(softmax_layer *l, network *net);

#ifdef GPU
void pull_softmax_layer_output(softmax_layer *l);
void forward_softmax_layer_gpu(softmax_layer *l, network *net);
void backward_softmax_layer_gpu(softmax_layer *l, network *net);
#endif

