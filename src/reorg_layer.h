#pragma once

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

void make_reorg_layer(layer *l, int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(layer *l);
void backward_reorg_layer(layer *l);

#ifdef GPU
void forward_reorg_layer_gpu(layer *l);
void backward_reorg_layer_gpu(layer *l);
#endif

