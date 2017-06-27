#pragma once

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef GPU
void forward_deconvolutional_layer_gpu(layer *l);
void backward_deconvolutional_layer_gpu(layer *l);
void update_deconvolutional_layer_gpu(layer *l, int batch, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(layer *l);
void pull_deconvolutional_layer(layer *l);
#endif

void make_deconvolutional_layer(layer *layer, int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(layer *l);
void update_deconvolutional_layer(layer *l, int batch, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(layer *l);

