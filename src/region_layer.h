#pragma once

#include "layer.h"
#include "network.h"

void make_region_layer(layer *l, int batch, int h, int w, int n, int classes, int coords);
void forward_region_layer(layer *l);
void backward_region_layer(layer *l);
void get_region_boxes(layer *l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh, int relative);
void resize_region_layer(layer *l, int w, int h);
void zero_objectness(layer *l);

#ifdef GPU
void forward_region_layer_gpu(layer *l);
void backward_region_layer_gpu(layer *l);
#endif

