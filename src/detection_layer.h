#pragma once

#include "layer.h"
#include "network.h"

typedef layer detection_layer;

void make_detection_layer(detection_layer *l, int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_detection_layer(detection_layer *l);
void backward_detection_layer(detection_layer *l);
void get_detection_boxes(layer *l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);

#ifdef GPU
void forward_detection_layer_gpu(detection_layer *l);
void backward_detection_layer_gpu(detection_layer *l);
#endif

