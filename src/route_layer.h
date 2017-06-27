#pragma once

#include "network.h"
#include "layer.h"

typedef layer route_layer;

void make_route_layer(route_layer *l, int batch, int n, int *input_layers, int *input_size);
void forward_route_layer(route_layer *l);
void backward_route_layer(route_layer *l);
void resize_route_layer(route_layer *l);

#ifdef GPU
void forward_route_layer_gpu(route_layer *l);
void backward_route_layer_gpu(route_layer *l);
#endif

