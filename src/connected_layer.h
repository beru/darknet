#pragma once

#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer connected_layer;

void make_connected_layer(connected_layer *l, int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

void forward_connected_layer(connected_layer *layer);
void backward_connected_layer(connected_layer *layer);
void update_connected_layer(connected_layer *layer, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(layer *l);
void statistics_connected_layer(layer *l);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer *layer);
void backward_connected_layer_gpu(connected_layer *layer);
void update_connected_layer_gpu(connected_layer *layer, int batch, float learning_rate, float momentum, float decay);
void push_connected_layer(connected_layer *layer);
void pull_connected_layer(connected_layer *layer);
#endif  // #ifdef GPU

