#pragma once

#include "image.h"
#include "layer.h"
#include "network.h"

void make_batchnorm_layer(layer *l, int batch, int w, int h, int c);
void forward_batchnorm_layer(layer *l);
void backward_batchnorm_layer(layer *l);

#ifdef GPU
void forward_batchnorm_layer_gpu(layer *l);
void backward_batchnorm_layer_gpu(layer *l);
void pull_batchnorm_layer(layer *l);
void push_batchnorm_layer(layer *l);
#endif

