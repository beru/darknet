#pragma once

#include "layer.h"
#include "network.h"

typedef layer cost_layer;

COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
void make_cost_layer(cost_layer *l, int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer(cost_layer *l);
void backward_cost_layer(cost_layer *l);
void resize_cost_layer(cost_layer *l, int inputs);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer *l);
void backward_cost_layer_gpu(cost_layer *l);
#endif

