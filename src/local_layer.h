#pragma once

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer local_layer;

#ifdef GPU
void forward_local_layer_gpu(local_layer *layer);
void backward_local_layer_gpu(local_layer *layer);
void update_local_layer_gpu(local_layer *layer, int batch, float learning_rate, float momentum, float decay);

void push_local_layer(local_layer *layer);
void pull_local_layer(local_layer *layer);
#endif

void make_local_layer(local_layer *l, int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);

void forward_local_layer(local_layer *layer);
void backward_local_layer(local_layer *layer);
void update_local_layer(local_layer *layer, int batch, float learning_rate, float momentum, float decay);

void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

