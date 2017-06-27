#pragma once

#include "layer.h"
#include "network.h"

void make_shortcut_layer(layer *l, int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer(layer *l);
void backward_shortcut_layer(layer *l);

#ifdef GPU
void forward_shortcut_layer_gpu(layer *l);
void backward_shortcut_layer_gpu(layer *l);
#endif

