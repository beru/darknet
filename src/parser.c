#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "utils.h"

typedef struct {
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char *type)
{
    if (strcmp(type, "[shortcut]") == 0) return SHORTCUT;
    if (strcmp(type, "[crop]") == 0) return CROP;
    if (strcmp(type, "[cost]") == 0) return COST;
    if (strcmp(type, "[detection]") == 0) return DETECTION;
    if (strcmp(type, "[region]") == 0) return REGION;
    if (strcmp(type, "[local]") == 0) return LOCAL;
    if (strcmp(type, "[conv]") == 0
            || strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]") == 0
            || strcmp(type, "[deconvolutional]") == 0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]") == 0) return ACTIVE;
    if (strcmp(type, "[net]") == 0
            || strcmp(type, "[network]") == 0) return NETWORK;
    if (strcmp(type, "[crnn]") == 0) return CRNN;
    if (strcmp(type, "[gru]") == 0) return GRU;
    if (strcmp(type, "[rnn]") == 0) return RNN;
    if (strcmp(type, "[conn]") == 0
            || strcmp(type, "[connected]") == 0) return CONNECTED;
    if (strcmp(type, "[max]") == 0
            || strcmp(type, "[maxpool]") == 0) return MAXPOOL;
    if (strcmp(type, "[reorg]") == 0) return REORG;
    if (strcmp(type, "[avg]") == 0
            || strcmp(type, "[avgpool]") == 0) return AVGPOOL;
    if (strcmp(type, "[dropout]") == 0) return DROPOUT;
    if (strcmp(type, "[lrn]") == 0
            || strcmp(type, "[normalization]") == 0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]") == 0) return BATCHNORM;
    if (strcmp(type, "[soft]") == 0
            || strcmp(type, "[softmax]") == 0) return SOFTMAX;
    if (strcmp(type, "[route]") == 0) return ROUTE;
    return BLANK;
}

void free_section(section *s)
{
    xplat_free(s->type);
    node *n = s->options->front;
    while (n) {
        kvp *pair = (kvp *)n->val;
        xplat_free(pair->key);
        xplat_free(pair);
        node *next = n->next;
        xplat_free(n);
        n = next;
    }
    xplat_free(s->options);
    xplat_free(s);
}

void parse_data(char *data, float *a, int n)
{
    if (!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for (int i = 0; i < n && !done; ++i) {
        while (*++next != '\0' && *next != ',');
        if (*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params {
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;

void parse_local(local_layer *l, list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int(options, "pad", 0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before local layer must output image.");
    }

    make_local_layer(l, batch, h, w, c, n, size, stride, pad, activation);
}

void parse_deconvolutional(layer *l, list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before deconvolutional layer must output image.");
    }
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    make_deconvolutional_layer(l,
                               batch, h, w, c, n,
                               size, stride, padding,
                               activation, batch_normalize, params.net.adam);
}


void parse_convolutional(convolutional_layer *l, list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before convolutional layer must output image.");
    }
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    make_convolutional_layer(l,
                             batch, h, w, c, n,
                             size, stride, padding,
                             activation, batch_normalize,
                             binary, xnor, params.net.adam,
                             params.net.train);
    l->flipped = option_find_int_quiet(options, "flipped", 0);
    l->dot = option_find_float_quiet(options, "dot", 0);
    if (params.net.adam) {
        l->B1 = params.net.B1;
        l->B2 = params.net.B2;
        l->eps = params.net.eps;
    }
}

void parse_crnn(layer *l, list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters", 1);
    int hidden_filters = option_find_int(options, "hidden_filters", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    make_crnn_layer(l,
                    params.batch, params.w, params.h, params.c,
                    hidden_filters, output_filters, params.time_steps,
                    activation, batch_normalize,
                    params.net.train);

    l->shortcut = option_find_int_quiet(options, "shortcut", 0);
}

void parse_rnn(layer *l, list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int hidden = option_find_int(options, "hidden", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int logistic = option_find_int_quiet(options, "logistic", 0);

    make_rnn_layer(l,
                   params.batch, params.inputs,
                   hidden, output, params.time_steps,
                   activation, batch_normalize, logistic);

    l->shortcut = option_find_int_quiet(options, "shortcut", 0);
}

void parse_gru(layer *l, list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    make_gru_layer(l, params.batch, params.inputs, output, params.time_steps, batch_normalize);
}

void parse_connected(connected_layer *l, list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    make_connected_layer(l,
                         params.batch, params.inputs,
                         output, activation, batch_normalize);
}

void parse_softmax(softmax_layer *l, list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups", 1);
    make_softmax_layer(l, params.batch, params.inputs, groups);
    l->temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) {
        l->softmax_tree = read_tree(tree_file);
    }
    l->w = params.w;
    l->h = params.h;
    l->c = params.c;
    l->spatial = option_find_float_quiet(options, "spatial", 0);
}

void parse_region(layer *l, list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    make_region_layer(l, params.batch, params.w, params.h, num, classes, coords);
    assert(l->outputs == params.inputs);

    l->log = option_find_int_quiet(options, "log", 0);
    l->sqrt = option_find_int_quiet(options, "sqrt", 0);

    l->softmax = option_find_int(options, "softmax", 0);
    l->background = option_find_int_quiet(options, "background", 0);
    l->max_boxes = option_find_int_quiet(options, "max",30);
    l->jitter = option_find_float(options, "jitter", .2);
    l->rescore = option_find_int_quiet(options, "rescore", 0);

    l->thresh = option_find_float(options, "thresh", .5);
    l->classfix = option_find_int_quiet(options, "classfix", 0);
    l->absolute = option_find_int_quiet(options, "absolute", 0);
    l->random = option_find_int_quiet(options, "random", 0);

    l->coord_scale = option_find_float(options, "coord_scale", 1);
    l->object_scale = option_find_float(options, "object_scale", 1);
    l->noobject_scale = option_find_float(options, "noobject_scale", 1);
    l->class_scale = option_find_float(options, "class_scale", 1);
    l->bias_match = option_find_int_quiet(options, "bias_match", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) {
        l->softmax_tree = read_tree(tree_file);
    }
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) {
        l->map = read_map(map_file);
    }

    char *a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        for (int i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (int i = 0; i < n; ++i) {
            float bias = atof(a);
            l->biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
}
void parse_detection(detection_layer *l, list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    make_detection_layer(l,
                         params.batch, params.inputs,
                         num, side, classes, coords, rescore);

    l->softmax = option_find_int(options, "softmax", 0);
    l->sqrt = option_find_int(options, "sqrt", 0);

    l->max_boxes = option_find_int_quiet(options, "max",30);
    l->coord_scale = option_find_float(options, "coord_scale", 1);
    l->forced = option_find_int(options, "forced", 0);
    l->object_scale = option_find_float(options, "object_scale", 1);
    l->noobject_scale = option_find_float(options, "noobject_scale", 1);
    l->class_scale = option_find_float(options, "class_scale", 1);
    l->jitter = option_find_float(options, "jitter", .2);
    l->random = option_find_int_quiet(options, "random", 0);
    l->reorg = option_find_int_quiet(options, "reorg", 0);
}

void parse_cost(cost_layer *l, list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale", 1);
    make_cost_layer(l, params.batch, params.inputs, type, scale);
    l->ratio =  option_find_float_quiet(options, "ratio", 0);
    l->thresh =  option_find_float_quiet(options, "thresh", 0);
}

void parse_crop(crop_layer *l, list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height", 1);
    int crop_width = option_find_int(options, "crop_width", 1);
    int flip = option_find_int(options, "flip", 0);
    float angle = option_find_float(options, "angle", 0);
    float saturation = option_find_float(options, "saturation", 1);
    float exposure = option_find_float(options, "exposure", 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before crop layer must output image.");
    }

    int noadjust = option_find_int_quiet(options, "noadjust", 0);

    make_crop_layer(l,
                    batch, h, w, c, crop_height, crop_width,
                    flip, angle, saturation, exposure);
    l->shift = option_find_float(options, "shift", 0);
    l->noadjust = noadjust;
}

void parse_reorg(layer *l, list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int reverse = option_find_int_quiet(options, "reverse", 0);
    int flatten = option_find_int_quiet(options, "flatten", 0);
    int extra = option_find_int_quiet(options, "extra", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before reorg layer must output image.");
    }

    make_reorg_layer(l, batch, w, h, c, stride, reverse, flatten, extra);
}

void parse_maxpool(maxpool_layer *l, list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before maxpool layer must output image.");
    }

    make_maxpool_layer(l, batch, h, w, c, size, stride, padding, params.net.train);
}

void parse_avgpool(avgpool_layer *l, list *options, size_params params)
{
    int batch, w, h, c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) {
        error("Layer before avgpool layer must output image.");
    }

    make_avgpool_layer(l, batch, w, h, c);
}

void parse_dropout(dropout_layer *l, list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    make_dropout_layer(l, params.batch, params.inputs, probability);
    l->out_w = params.w;
    l->out_h = params.h;
    l->out_c = params.c;
}

void parse_normalization(layer *l, list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    make_normalization_layer(l,
                             params.batch, params.w, params.h, params.c,
                             size, alpha, beta, kappa);
}

void parse_batchnorm(layer *l, list *options, size_params params)
{
    make_batchnorm_layer(l, params.batch, params.w, params.h, params.c);
}

void parse_shortcut(layer *l, list *options, size_params params, network *net)
{
    char *from_option = option_find(options, "from");   
    int index = atoi(from_option);
    if (index < 0) {
        index = params.index + index;
    }

    int batch = params.batch;
    layer *from = &net->layers[index];

    make_shortcut_layer(l,
                        batch, index, params.w, params.h, params.c,
                        from->out_w, from->out_h, from->out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    l->activation = activation;
}

void parse_activation(layer *l, list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    make_activation_layer(l, params.batch, params.inputs, activation);

    l->out_h = params.h;
    l->out_w = params.w;
    l->out_c = params.c;
    l->h = params.h;
    l->w = params.w;
    l->c = params.c;
}

void parse_route(route_layer *l, list *options, size_params params, network *net)
{
    char *opt_layers = option_find(options, "layers");   
    int len = strlen(opt_layers);
    if (!opt_layers) {
        error("Route Layer must specify input layers");
    }
    int n = 1;
    for (int i = 0; i < len; ++i) {
        if (opt_layers[i] == ',') ++n;
    }

    int *layers = xplat_malloc(n, sizeof(int));
    int *sizes = xplat_malloc(n, sizeof(int));
    for (int i = 0; i < n; ++i) {
        int index = atoi(opt_layers);
        opt_layers = strchr(opt_layers, ',') + 1;
        if (index < 0) {
            index = params.index + index;
        }
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    make_route_layer(l, batch, n, layers, sizes);

    convolutional_layer *first = &net->layers[layers[0]];
    l->out_w = first->out_w;
    l->out_h = first->out_h;
    l->out_c = first->out_c;
    for (int i = 1; i < n; ++i) {
        int index = layers[i];
        convolutional_layer *next = &net->layers[index];
        if (next->out_w == first->out_w && next->out_h == first->out_h) {
            l->out_c += next->out_c;
        }else {
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random") == 0) return RANDOM;
    if (strcmp(s, "poly") == 0) return POLY;
    if (strcmp(s, "constant") == 0) return CONSTANT;
    if (strcmp(s, "step") == 0) return STEP;
    if (strcmp(s, "exp") == 0) return EXP;
    if (strcmp(s, "sigmoid") == 0) return SIG;
    if (strcmp(s, "steps") == 0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);
    net->notruth = option_find_int_quiet(options, "notruth", 0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    net->adam = option_find_int_quiet(options, "adam", 0);
    if (net->adam) {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .00000001);
    }

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w * 2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->center = option_find_int_quiet(options, "center", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c)) {
        error("No input parameters supplied");
    }

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if (net->policy == STEP) {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }else if (net->policy == STEPS) {
        char *l = option_find(options, "steps");   
        char *p = option_find(options, "scales");   
        if (!l || !p) {
            error("STEPS policy must have steps and scales in cfg file");
        }

        int len = strlen(l);
        int n = 1;
        for (int i = 0; i < len; ++i) {
            if (l[i] == ',') ++n;
        }
        int *steps = xplat_malloc(n, sizeof(int));
        float *scales = xplat_malloc(n, sizeof(float));
        for (int i = 0; i < n; ++i) {
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }else if (net->policy == EXP) {
        net->gamma = option_find_float(options, "gamma", 1);
    }else if (net->policy == SIG) {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }else if (net->policy == POLY || net->policy == RANDOM) {
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]") == 0
            || strcmp(s->type, "[network]") == 0);
}

void parse_network_cfg(network *net, char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n) {
        error("Config file has no sections");
    }
    make_network(net, sections->size - 1);
#ifdef GPU
    net->gpu_index = gpu_index;
#endif
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if (!is_network(s)) {
        error("First section must be [net] or [network]");
    }
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = *net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while (n) {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer *l = &net->layers[count];
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if (lt == CONVOLUTIONAL) {
            parse_convolutional(l, options, params);
        }else if (lt == DECONVOLUTIONAL) {
            parse_deconvolutional(l, options, params);
        }else if (lt == LOCAL) {
            parse_local(l, options, params);
        }else if (lt == ACTIVE) {
            parse_activation(l, options, params);
        }else if (lt == RNN) {
            parse_rnn(l, options, params);
        }else if (lt == GRU) {
            parse_gru(l, options, params);
        }else if (lt == CRNN) {
            parse_crnn(l, options, params);
        }else if (lt == CONNECTED) {
            parse_connected(l, options, params);
        }else if (lt == CROP) {
            parse_crop(l, options, params);
        }else if (lt == COST) {
            parse_cost(l, options, params);
        }else if (lt == REGION) {
            parse_region(l, options, params);
        }else if (lt == DETECTION) {
            parse_detection(l, options, params);
        }else if (lt == SOFTMAX) {
            parse_softmax(l, options, params);
            net->hierarchy = l->softmax_tree;
        }else if (lt == NORMALIZATION) {
            parse_normalization(l, options, params);
        }else if (lt == BATCHNORM) {
            parse_batchnorm(l, options, params);
        }else if (lt == MAXPOOL) {
            parse_maxpool(l, options, params);
        }else if (lt == REORG) {
            parse_reorg(l, options, params);
        }else if (lt == AVGPOOL) {
            parse_avgpool(l, options, params);
        }else if (lt == ROUTE) {
            parse_route(l, options, params, net);
        }else if (lt == SHORTCUT) {
            parse_shortcut(l, options, params, net);
        }else if (lt == DROPOUT) {
            parse_dropout(l, options, params);
            l->output = net->layers[count - 1].output;
            l->delta = net->layers[count - 1].delta;
#ifdef GPU
            l->output_gpu = net->layers[count - 1].output_gpu;
            l->delta_gpu = net->layers[count - 1].delta_gpu;
#endif
        }else {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l->truth = option_find_int_quiet(options, "truth", 0);
        l->onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l->stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l->dontload = option_find_int_quiet(options, "dontload", 0);
        l->dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l->learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l->smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        if (l->workspace_size > workspace_size) {
            workspace_size = l->workspace_size;
        }
        free_section(s);
        n = n->next;
        ++count;
        if (n) {
            params.h = l->out_h;
            params.w = l->out_w;
            params.c = l->out_c;
            params.inputs = l->outputs;
        }
    }   
    free_list(sections);
    layer *out = get_network_output_layer(net);
    net->outputs = out->outputs;
    net->truths = out->outputs;
    if (net->layers[net->n - 1].truths) {
        net->truths = net->layers[net->n - 1].truths;
    }
    net->output = out->output;
    //net->input = xplat_malloc(net->inputs * net->batch, sizeof(float));
    //net->truth = xplat_malloc(net->truths * net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out->output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs * net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths * net->batch);
#endif
    if (workspace_size) {
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if (gpu_index >= 0) {
            net->workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
        }else {
            net->workspace = xplat_malloc(1, workspace_size);
        }
#else
        net->workspace = xplat_malloc(1, workspace_size);
#endif
    }
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0) {
        file_error(filename);
    }
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while ((line = fgetl(file)) != 0) {
        ++ nu;
        strip(line);
        switch (line[0]) {
        case '[':
            current = xplat_malloc(1, sizeof(section));
            list_insert(options, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            xplat_free(line);
            break;
        default:
            if (!read_option(line, current->options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n",
                        nu, line);
                xplat_free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_convolutional_layer(&l);
    }
#endif
    binarize_weights(l.weights, l.n, l.c * l.size * l.size, l.binary_weights);
    int size = l.c * l.size * l.size;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize) {
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for (int i = 0; i < l.n; ++i) {
        float mean = l.binary_weights[i * size];
        if (mean < 0) {
            mean = -mean;
        }
        fwrite(&mean, sizeof(float), 1, fp);
        for (int j = 0; j < size/8; ++j) {
            int index = i * size + j * 8;
            unsigned char c = 0;
            for (int k = 0; k < 8; ++k) {
                if (j * 8 + k >= size) {
                    break;
                }
                if (l.binary_weights[index + k] > 0) {
                    c = (c | 1<<k);
                }
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer *l, FILE *fp)
{
    if (l->binary) {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if (gpu_index >= 0) {
        pull_convolutional_layer(l);
    }
#endif
    int num = l->n * l->c * l->size * l->size;
    fwrite(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize) {
        fwrite(l->scales, sizeof(float), l->n, fp);
        fwrite(l->rolling_mean, sizeof(float), l->n, fp);
        fwrite(l->rolling_variance, sizeof(float), l->n, fp);
    }
    fwrite(l->weights, sizeof(float), num, fp);
    if (l->adam) {
        //fwrite(l.m, sizeof(float), num, fp);
        //fwrite(l.v, sizeof(float), num, fp);
    }
}

void save_batchnorm_weights(layer *l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l->scales, sizeof(float), l->c, fp);
    fwrite(l->rolling_mean, sizeof(float), l->c, fp);
    fwrite(l->rolling_variance, sizeof(float), l->c, fp);
}

void save_connected_weights(layer *l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_connected_layer(l);
    }
#endif
    fwrite(l->biases, sizeof(float), l->outputs, fp);
    fwrite(l->weights, sizeof(float), l->outputs * l->inputs, fp);
    if (l->batch_normalize) {
        fwrite(l->scales, sizeof(float), l->outputs, fp);
        fwrite(l->rolling_mean, sizeof(float), l->outputs, fp);
        fwrite(l->rolling_variance, sizeof(float), l->outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0) {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        file_error(filename);
    }

    int major = 0;
    int minor = 1;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(int), 1, fp);

    for (int i = 0; i < net->n && i < cutoff; ++i) {
        layer *l = &net->layers[i];
        if (l->type == CONVOLUTIONAL || l->type == DECONVOLUTIONAL) {
            save_convolutional_weights(l, fp);
        } if (l->type == CONNECTED) {
            save_connected_weights(l, fp);
        } if (l->type == BATCHNORM) {
            save_batchnorm_weights(l, fp);
        } if (l->type == RNN) {
            save_connected_weights(l->input_layer, fp);
            save_connected_weights(l->self_layer, fp);
            save_connected_weights(l->output_layer, fp);
        } if (l->type == GRU) {
            save_connected_weights(l->input_z_layer, fp);
            save_connected_weights(l->input_r_layer, fp);
            save_connected_weights(l->input_h_layer, fp);
            save_connected_weights(l->state_z_layer, fp);
            save_connected_weights(l->state_r_layer, fp);
            save_connected_weights(l->state_h_layer, fp);
        } if (l->type == CRNN) {
            save_convolutional_weights(l->input_layer, fp);
            save_convolutional_weights(l->self_layer, fp);
            save_convolutional_weights(l->output_layer, fp);
        } if (l->type == LOCAL) {
#ifdef GPU
            if (gpu_index >= 0) {
                pull_local_layer(l);
            }
#endif
            int locations = l->out_w * l->out_h;
            int size = l->size * l->size * l->c * l->n * locations;
            fwrite(l->biases, sizeof(float), l->outputs, fp);
            fwrite(l->weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = xplat_malloc(rows * cols, sizeof(float));
    int x, y;
    for (x = 0; x < rows; ++x) {
        for (y = 0; y < cols; ++y) {
            transpose[y * rows + x] = a[x * cols + y];
        }
    }
    memcpy(a, transpose, rows * cols * sizeof(float));
    xplat_free(transpose);
}

void load_connected_weights(layer *l, FILE *fp, int transpose)
{
    fread(l->biases, sizeof(float), l->outputs, fp);
    fread(l->weights, sizeof(float), l->outputs * l->inputs, fp);
    if (transpose) {
        transpose_matrix(l->weights, l->inputs, l->outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l->batch_normalize && (!l->dontloadscales)) {
        fread(l->scales, sizeof(float), l->outputs, fp);
        fread(l->rolling_mean, sizeof(float), l->outputs, fp);
        fread(l->rolling_variance, sizeof(float), l->outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if (gpu_index >= 0) {
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer *l, FILE *fp)
{
    fread(l->scales, sizeof(float), l->c, fp);
    fread(l->rolling_mean, sizeof(float), l->c, fp);
    fread(l->rolling_variance, sizeof(float), l->c, fp);
#ifdef GPU
    if (gpu_index >= 0) {
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)) {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c * l.size * l.size;
    for (int i = 0; i < l.n; ++i) {
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for (int j = 0; j < size / 8; ++j) {
            int index = i * size + j * 8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for (int k = 0; k < 8; ++k) {
                if (j * 8 + k >= size) {
                    break;
                }
                l.weights[index + k] = (c & 1 << k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if (gpu_index >= 0) {
        push_convolutional_layer(&l);
    }
#endif
}

void load_convolutional_weights(layer *l, FILE *fp)
{
    if (l->binary) {
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l->n * l->c * l->size * l->size;
    fread(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize && (!l->dontloadscales)) {
        fread(l->scales, sizeof(float), l->n, fp);
        fread(l->rolling_mean, sizeof(float), l->n, fp);
        fread(l->rolling_variance, sizeof(float), l->n, fp);
        if (0) {
            for (int i = 0; i < l->n; ++i) {
                printf("%g, ", l->rolling_mean[i]);
            }
            printf("\n");
            for (int i = 0; i < l->n; ++i) {
                printf("%g, ", l->rolling_variance[i]);
            }
            printf("\n");
        }
        if (0) {
            fill_cpu(l->n, 0, l->rolling_mean, 1);
            fill_cpu(l->n, 0, l->rolling_variance, 1);
        }
    }
    fread(l->weights, sizeof(float), num, fp);
    if (l->adam) {
        //fread(l.m, sizeof(float), num, fp);
        //fread(l.v, sizeof(float), num, fp);
    }
    //if (l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l->flipped) {
        transpose_matrix(l->weights, l->c * l->size * l->size, l->n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if (gpu_index >= 0) {
        push_convolutional_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0) {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        file_error(filename);
    }

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(net->seen, sizeof(int), 1, fp);
    int transpose = (major > 1000) || (minor > 1000);

    for (int i = start; i < net->n && i < cutoff; ++i) {
        layer *l = &net->layers[i];
        if (l->dontload) {
            continue;
        }
        if (l->type == CONVOLUTIONAL || l->type == DECONVOLUTIONAL) {
            load_convolutional_weights(l, fp);
        }
        if (l->type == CONNECTED) {
            load_connected_weights(l, fp, transpose);
        }
        if (l->type == BATCHNORM) {
            load_batchnorm_weights(l, fp);
        }
        if (l->type == CRNN) {
            load_convolutional_weights(l->input_layer, fp);
            load_convolutional_weights(l->self_layer, fp);
            load_convolutional_weights(l->output_layer, fp);
        }
        if (l->type == RNN) {
            load_connected_weights(l->input_layer, fp, transpose);
            load_connected_weights(l->self_layer, fp, transpose);
            load_connected_weights(l->output_layer, fp, transpose);
        }
        if (l->type == GRU) {
            load_connected_weights(l->input_z_layer, fp, transpose);
            load_connected_weights(l->input_r_layer, fp, transpose);
            load_connected_weights(l->input_h_layer, fp, transpose);
            load_connected_weights(l->state_z_layer, fp, transpose);
            load_connected_weights(l->state_r_layer, fp, transpose);
            load_connected_weights(l->state_h_layer, fp, transpose);
        }
        if (l->type == LOCAL) {
            int locations = l->out_w * l->out_h;
            int size = l->size * l->size * l->c * l->n * locations;
            fread(l->biases, sizeof(float), l->outputs, fp);
            fread(l->weights, sizeof(float), size, fp);
#ifdef GPU
            if (gpu_index >= 0) {
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

