#pragma once

#include "image.h"
#include "layer.h"
#include "network.h"

void make_normalization_layer(layer *l, int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer *l, int h, int w);
void forward_normalization_layer(layer *l, network *net);
void backward_normalization_layer(layer *l, network *net);
void visualize_normalization_layer(layer *l, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(layer *l, network *net);
void backward_normalization_layer_gpu(layer *l, network *net);
#endif

