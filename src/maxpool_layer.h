#pragma once

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer maxpool_layer;

image get_maxpool_image(maxpool_layer *l);
void make_maxpool_layer(maxpool_layer *l, int batch, int h, int w, int c, int size, int stride, int padding, int train);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(maxpool_layer *l, network *net);
void backward_maxpool_layer(maxpool_layer *l, network *net);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer *l, network *net);
void backward_maxpool_layer_gpu(maxpool_layer *l, network *net);
#endif

