#pragma once

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer *l);
void make_avgpool_layer(avgpool_layer *l, int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(avgpool_layer *l);
void backward_avgpool_layer(avgpool_layer *l);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer *l);
void backward_avgpool_layer_gpu(avgpool_layer *l);
#endif

