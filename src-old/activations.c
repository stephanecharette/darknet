#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case GELU:
            return "gelu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}


float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case GELU:
            return gelu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case REVLEAKY:
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR) {}
    else if (a == LEAKY) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == LOGISTIC) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}




void gradient_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float d = delta[index];
                d = d * grad;
                delta[index] = d;
            }
        }
    }
}

void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                if (x[index] > 0) {
                    float d = delta[index];
                    d = d * grad;
                    delta[index] = d;
                }
            }
        }
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case RELU6:
            return relu6_gradient(x);
        case NORM_CHAN:
            //return relu_gradient(x);
        case NORM_CHAN_SOFTMAX_MAXVAL:
            //...
        case NORM_CHAN_SOFTMAX:
            printf(" Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient \n");
            exit(0);
            return 0;
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case GELU:
            return gelu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case REVLEAKY:
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}

// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(const float *x, const int n, const float * sigmoid, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float swish = x[i];
        delta[i] *= swish + sigmoid[i]*(1 - swish);
    }
}

// https://github.com/digantamisra98/Mish
void gradient_array_mish(const int n, const float * activation_input, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float inp = activation_input[i];
        const float sp = softplus_activate(inp, MISH_THRESHOLD);
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;


        //float x = activation_input[i];
        //float d = 2 * expf(x) + expf(2 * x) + 2;
        //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
        //float derivative = expf(x) * w / (d * d);
        //delta[i] *= derivative;
    }
}

static float hard_mish_yashas_grad(float x)
{
    if (x > 0)
        return 1;
    if (x > -2)
        return x + 1;
    return 0;
}

void gradient_array_hard_mish(const int n, const float * activation_input, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float inp = activation_input[i];
        delta[i] *= hard_mish_yashas_grad(inp);
    }
}
