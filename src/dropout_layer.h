#pragma once

#include "layer.h"
#include "network.h"

typedef layer dropout_layer;

void make_dropout_layer(dropout_layer *l, int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer *l);
void backward_dropout_layer(dropout_layer *l);
void resize_dropout_layer(dropout_layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer *l);
void backward_dropout_layer_gpu(dropout_layer *l);
#endif
