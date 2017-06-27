#pragma once

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

void make_rnn_layer(layer *l, int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);

void forward_rnn_layer(layer *l);
void backward_rnn_layer(layer *l);
void update_rnn_layer(layer *l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_rnn_layer_gpu(layer *l);
void backward_rnn_layer_gpu(layer *l);
void update_rnn_layer_gpu(layer *l, int batch, float learning_rate, float momentum, float decay);
void push_rnn_layer(layer *l);
void pull_rnn_layer(layer *l);
#endif

