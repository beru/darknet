#pragma once

#include "activations.h"
#include "layer.h"
#include "network.h"

void make_gru_layer(layer *l, int batch, int inputs, int outputs, int steps, int batch_normalize);

void forward_gru_layer(layer *l);
void backward_gru_layer(layer *l);
void update_gru_layer(layer *l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_gru_layer_gpu(layer *l);
void backward_gru_layer_gpu(layer *l);
void update_gru_layer_gpu(layer *l, int batch, float learning_rate, float momentum, float decay);
void push_gru_layer(layer *l);
void pull_gru_layer(layer *l);
#endif

