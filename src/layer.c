#include "layer.h"
#include "cuda.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if (l.type == DROPOUT) {
        if (l.rand)           xplat_free(l.rand);
#ifdef GPU
        if (l.rand_gpu)       cuda_free(l.rand_gpu);
#endif
        return;
    }
    if (l.cweights)           xplat_free(l.cweights);
    if (l.indexes)            xplat_free(l.indexes);
    if (l.input_layers)       xplat_free(l.input_layers);
    if (l.input_sizes)        xplat_free(l.input_sizes);
    if (l.map)                xplat_free(l.map);
    if (l.rand)               xplat_free(l.rand);
    if (l.cost)               xplat_free(l.cost);
    if (l.state)              xplat_free(l.state);
    if (l.prev_state)         xplat_free(l.prev_state);
    if (l.forgot_state)       xplat_free(l.forgot_state);
    if (l.forgot_delta)       xplat_free(l.forgot_delta);
    if (l.state_delta)        xplat_free(l.state_delta);
    if (l.concat)             xplat_free(l.concat);
    if (l.concat_delta)       xplat_free(l.concat_delta);
    if (l.binary_weights)     xplat_free(l.binary_weights);
    if (l.biases)             xplat_free(l.biases);
    if (l.bias_updates)       xplat_free(l.bias_updates);
    if (l.scales)             xplat_free(l.scales);
    if (l.scale_updates)      xplat_free(l.scale_updates);
    if (l.weights)            xplat_free(l.weights);
    if (l.weight_updates)     xplat_free(l.weight_updates);
    if (l.delta)              xplat_free(l.delta);
    if (l.output)             xplat_free(l.output);
    if (l.squared)            xplat_free(l.squared);
    if (l.norms)              xplat_free(l.norms);
    if (l.spatial_mean)       xplat_free(l.spatial_mean);
    if (l.mean)               xplat_free(l.mean);
    if (l.variance)           xplat_free(l.variance);
    if (l.mean_delta)         xplat_free(l.mean_delta);
    if (l.variance_delta)     xplat_free(l.variance_delta);
    if (l.rolling_mean)       xplat_free(l.rolling_mean);
    if (l.rolling_variance)   xplat_free(l.rolling_variance);
    if (l.x)                  xplat_free(l.x);
    if (l.x_norm)             xplat_free(l.x_norm);
    if (l.m)                  xplat_free(l.m);
    if (l.v)                  xplat_free(l.v);
    if (l.z_cpu)              xplat_free(l.z_cpu);
    if (l.r_cpu)              xplat_free(l.r_cpu);
    if (l.h_cpu)              xplat_free(l.h_cpu);
    if (l.binary_input)       xplat_free(l.binary_input);
 
#ifdef GPU
    if (l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if (l.z_gpu)                   cuda_free(l.z_gpu);
    if (l.r_gpu)                   cuda_free(l.r_gpu);
    if (l.h_gpu)                   cuda_free(l.h_gpu);
    if (l.m_gpu)                   cuda_free(l.m_gpu);
    if (l.v_gpu)                   cuda_free(l.v_gpu);
    if (l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if (l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if (l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if (l.state_gpu)               cuda_free(l.state_gpu);
    if (l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if (l.gate_gpu)                cuda_free(l.gate_gpu);
    if (l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if (l.save_gpu)                cuda_free(l.save_gpu);
    if (l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if (l.concat_gpu)              cuda_free(l.concat_gpu);
    if (l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if (l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if (l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if (l.mean_gpu)                cuda_free(l.mean_gpu);
    if (l.variance_gpu)            cuda_free(l.variance_gpu);
    if (l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if (l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if (l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if (l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if (l.x_gpu)                   cuda_free(l.x_gpu);
    if (l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if (l.weights_gpu)             cuda_free(l.weights_gpu);
    if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if (l.biases_gpu)              cuda_free(l.biases_gpu);
    if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if (l.scales_gpu)              cuda_free(l.scales_gpu);
    if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if (l.output_gpu)              cuda_free(l.output_gpu);
    if (l.delta_gpu)               cuda_free(l.delta_gpu);
    if (l.rand_gpu)                cuda_free(l.rand_gpu);
    if (l.squared_gpu)             cuda_free(l.squared_gpu);
    if (l.norms_gpu)               cuda_free(l.norms_gpu);
#endif // #ifdef GPU
}
