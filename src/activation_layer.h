#pragma once

#include "activations.h"
#include "layer.h"
#include "network.h"

void make_activation_layer(layer *l, int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer *l, network *net);
void backward_activation_layer(layer *l, network *net);

#ifdef GPU
void forward_activation_layer_gpu(layer *l, network *net);
void backward_activation_layer_gpu(layer *l, network *net);
#endif


