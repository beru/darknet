#pragma once

#include "network.h"
#include "layer.h"

typedef layer route_layer;

void make_route_layer(route_layer *l, int batch, int n, int *input_layers, int *input_size);
void forward_route_layer(route_layer *l, network *net);
void backward_route_layer(route_layer *l, network *net);
void resize_route_layer(route_layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(route_layer *l, network *net);
void backward_route_layer_gpu(route_layer *l, network *net);
#endif

