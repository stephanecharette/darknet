#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "activation_layer.h"
#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "lstm_layer.h"
#include "conv_lstm_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "scale_channels_layer.h"
#include "sam_layer.h"
#include "softmax_layer.h"
#include "utils.h"
#include "upsample_layer.h"
#include "version.h"
#include "yolo_layer.h"
#include "gaussian_yolo_layer.h"
#include "representation_layer.h"

void empty_func(dropout_layer l, network_state state) {
    //l.output_gpu = state.input;
}

list *read_cfg(char *filename);


void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int train;
    network net;
} size_params;

local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.", DARKNET_LOC);

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}


layer parse_crnn(list *options, size_params params)
{
    int size = option_find_int_quiet(options, "size", 3);
    int stride = option_find_int_quiet(options, "stride", 1);
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    int output_filters = option_find_int(options, "output",1);
    int hidden_filters = option_find_int(options, "hidden",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    layer l = make_crnn_layer(params.batch, params.h, params.w, params.c, hidden_filters, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, xnor, params.train);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int hidden = option_find_int(options, "hidden",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int logistic = option_find_int_quiet(options, "logistic", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, hidden, output, params.time_steps, activation, batch_normalize, logistic);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize);

    return l;
}

layer parse_conv_lstm(list *options, size_params params)
{
    // a ConvLSTM with a larger transitional kernel should be able to capture faster motions
    int size = option_find_int_quiet(options, "size", 3);
    int stride = option_find_int_quiet(options, "stride", 1);
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    int output_filters = option_find_int(options, "output", 1);
    int groups = option_find_int_quiet(options, "groups", 1);
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int peephole = option_find_int_quiet(options, "peephole", 0);
    int bottleneck = option_find_int_quiet(options, "bottleneck", 0);

    layer l = make_conv_lstm_layer(params.batch, params.h, params.w, params.c, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, peephole, xnor, bottleneck, params.train);

    l.state_constrain = option_find_int_quiet(options, "state_constrain", params.time_steps * 32);
    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    char *lstm_activation_s = option_find_str(options, "lstm_activation", "tanh");
    l.lstm_activation = get_activation(lstm_activation_s);
    l.time_normalizer = option_find_float_quiet(options, "time_normalizer", 1.0);

    return l;
}

layer parse_history(list *options, size_params params)
{
    int history_size = option_find_int(options, "history_size", 4);
    layer l = make_history_layer(params.batch, params.h, params.w, params.c, history_size, params.time_steps, params.train);
    return l;
}

connected_layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    connected_layer layer = make_connected_layer(params.batch, 1, params.inputs, output, activation, batch_normalize);

    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, "groups", 1);
	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
	layer.temperature = option_find_float_quiet(options, "temperature", 1);
	char *tree_file = option_find_str(options, "tree", 0);
	if (tree_file) layer.softmax_tree = read_tree(tree_file);
	layer.w = params.w;
	layer.h = params.h;
	layer.c = params.c;
	layer.spatial = option_find_float_quiet(options, "spatial", 0);
	layer.noloss = option_find_int_quiet(options, "noloss", 0);
	return layer;
}

contrastive_layer parse_contrastive(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 1000);
    layer *yolo_layer = NULL;
    int yolo_layer_id = option_find_int_quiet(options, "yolo_layer", 0);
    if (yolo_layer_id < 0) yolo_layer_id = params.index + yolo_layer_id;
    if(yolo_layer_id != 0) yolo_layer = params.net.layers + yolo_layer_id;
    if (yolo_layer->type != YOLO) {
        printf(" Error: [contrastive] layer should point to the [yolo] layer instead of %d layer! \n", yolo_layer_id);
        getchar();
        exit(0);
    }

    contrastive_layer layer = make_contrastive_layer(params.batch, params.w, params.h, params.c, classes, params.inputs, yolo_layer);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    layer.steps = params.time_steps;
    layer.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    layer.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    layer.contrastive_neg_max = option_find_int_quiet(options, "contrastive_neg_max", 3);
    return layer;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int*)xcalloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

float *get_classes_multipliers(char *cpc, const int classes, const float max_delta)
{
    float *classes_multipliers = NULL;
    if (cpc) {
        int classes_counters = classes;
        int *counters_per_class = parse_yolo_mask(cpc, &classes_counters);
        if (classes_counters != classes) {
            printf(" number of values in counters_per_class = %d doesn't match with classes = %d \n", classes_counters, classes);
            exit(0);
        }
        float max_counter = 0;
        int i;
        for (i = 0; i < classes_counters; ++i) {
            if (counters_per_class[i] < 1) counters_per_class[i] = 1;
            if (max_counter < counters_per_class[i]) max_counter = counters_per_class[i];
        }
        classes_multipliers = (float *)calloc(classes_counters, sizeof(float));
        for (i = 0; i < classes_counters; ++i) {
            classes_multipliers[i] = max_counter / counters_per_class[i];
            if(classes_multipliers[i] > max_delta) classes_multipliers[i] = max_delta;
        }
        free(counters_per_class);
        printf(" classes_multipliers: ");
        for (i = 0; i < classes_counters; ++i) printf("%.1f, ", classes_multipliers[i]);
        printf("\n");
    }
    return classes_multipliers;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;
    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    int max_boxes = option_find_int_quiet(options, "max", 200);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);

    l.show_details = option_find_int_quiet(options, "show_details", 1);
    l.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    char *cpc = option_find_str(options, "counters_per_class", 0);
    l.classes_multipliers = get_classes_multipliers(cpc, classes, l.max_delta);

    l.label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    l.scale_x_y = option_find_float_quiet(options, "scale_x_y", 1);
    l.objectness_smooth = option_find_int_quiet(options, "objectness_smooth", 0);
    l.new_coords = option_find_int_quiet(options, "new_coords", 0);
    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 0.75);
    l.obj_normalizer = option_find_float_quiet(options, "obj_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    l.delta_normalizer = option_find_float_quiet(options, "delta_normalizer", 1);
    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //  "iou");

    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;
    fprintf(stderr, "[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale_x_y: %2.2f\n",
        iou_loss, l.iou_loss, l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y);

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);
    l.focal_loss = option_find_int_quiet(options, "focal_loss", 0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.iou_thresh = option_find_float_quiet(options, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
    l.random = option_find_float_quiet(options, "random", 0);

    l.track_history_size = option_find_int_quiet(options, "track_history_size", 5);
    l.sim_thresh = option_find_float_quiet(options, "sim_thresh", 0.8);
    l.dets_for_track = option_find_int_quiet(options, "dets_for_track", 1);
    l.dets_for_show = option_find_int_quiet(options, "dets_for_show", 1);
    l.track_ciou_norm = option_find_float_quiet(options, "track_ciou_norm", 0.01);
    int embedding_layer_id = option_find_int_quiet(options, "embedding_layer", 999999);
    if (embedding_layer_id < 0) embedding_layer_id = params.index + embedding_layer_id;
    if (embedding_layer_id != 999999) {
        printf(" embedding_layer_id = %d, ", embedding_layer_id);
        layer le = params.net.layers[embedding_layer_id];
        l.embedding_layer_id = embedding_layer_id;
        l.embedding_output = (float*)xcalloc(le.batch * le.outputs, sizeof(float));
        l.embedding_size = le.n / l.n;
        printf(" embedding_size = %d \n", l.embedding_size);
        if (le.n % l.n != 0) {
            printf(" Warning: filters=%d number in embedding_layer=%d isn't divisable by number of anchors %d \n", le.n, embedding_layer_id, l.n);
            getchar();
        }
    }

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n && i < total*2; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}


int *parse_gaussian_yolo_mask(char *a, int *num) // Gaussian_YOLOv3
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int *)calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}


layer parse_gaussian_yolo(list *options, size_params params) // Gaussian_YOLOv3
{
    int classes = option_find_int(options, "classes", 20);
    int max_boxes = option_find_int_quiet(options, "max", 200);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_gaussian_yolo_mask(a, &num);
    layer l = make_gaussian_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [Gaussian_yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);
    l.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    char *cpc = option_find_str(options, "counters_per_class", 0);
    l.classes_multipliers = get_classes_multipliers(cpc, classes, l.max_delta);

    l.label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    l.scale_x_y = option_find_float_quiet(options, "scale_x_y", 1);
    l.objectness_smooth = option_find_int_quiet(options, "objectness_smooth", 0);
    l.uc_normalizer = option_find_float_quiet(options, "uc_normalizer", 1.0);
    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 0.75);
    l.obj_normalizer = option_find_float_quiet(options, "obj_normalizer", 1.0);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    l.delta_normalizer = option_find_float_quiet(options, "delta_normalizer", 1);
    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //  "iou");

    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else if (strcmp(nms_kind, "cornersnms") == 0) l.nms_kind = CORNERS_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    char *yolo_point = option_find_str_quiet(options, "yolo_point", "center");
    if (strcmp(yolo_point, "left_top") == 0) l.yolo_point = YOLO_LEFT_TOP;
    else if (strcmp(yolo_point, "right_bottom") == 0) l.yolo_point = YOLO_RIGHT_BOTTOM;
    else l.yolo_point = YOLO_CENTER;

    fprintf(stderr, "[Gaussian_yolo] iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale: %2.2f, point: %d\n",
        iou_loss, l.iou_loss, l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y, l.yolo_point);

    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.iou_thresh = option_find_float_quiet(options, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
    l.random = option_find_float_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);
    int max_boxes = option_find_int_quiet(options, "max", 200);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or num= in [region]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.focal_loss = option_find_int_quiet(options, "focal_loss", 0);
    //l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_float_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n && i < num*2; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",200);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.resize = option_find_float_quiet(options, "resize", 1.0);
    layer.random = option_find_float_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.", DARKNET_LOC);

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.", DARKNET_LOC);

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse);
    return layer;
}

layer parse_reorg_old(list *options, size_params params)
{
    printf("\n reorg_old \n");
    int stride = option_find_int(options, "stride", 1);
    int reverse = option_find_int_quiet(options, "reverse", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before reorg layer must output image.", DARKNET_LOC);

    layer layer = make_reorg_old_layer(batch, w, h, c, stride, reverse);
    return layer;
}

maxpool_layer parse_local_avgpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int stride_x = option_find_int_quiet(options, "stride_x", stride);
    int stride_y = option_find_int_quiet(options, "stride_y", stride);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", size - 1);
    int maxpool_depth = 0;
    int out_channels = 1;
    int antialiasing = 0;
    const int avgpool = 1;

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before [local_avgpool] layer must output image.", DARKNET_LOC);

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", stride);
    int stride_y = option_find_int_quiet(options, "stride_y", stride);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    int maxpool_depth = option_find_int_quiet(options, "maxpool_depth", 0);
    int out_channels = option_find_int_quiet(options, "out_channels", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    const int avgpool = 0;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before [maxpool] layer must output image.", DARKNET_LOC);

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    layer.maxpool_zero_nonmax = option_find_int_quiet(options, "maxpool_zero_nonmax", 0);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.", DARKNET_LOC);

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .2);
    int dropblock = option_find_int_quiet(options, "dropblock", 0);
    float dropblock_size_rel = option_find_float_quiet(options, "dropblock_size_rel", 0);
    int dropblock_size_abs = option_find_float_quiet(options, "dropblock_size_abs", 0);
    if (dropblock_size_abs > params.w || dropblock_size_abs > params.h) {
        printf(" [dropout] - dropblock_size_abs = %d that is bigger than layer size %d x %d \n", dropblock_size_abs, params.w, params.h);
        dropblock_size_abs = min_val_cmp(params.w, params.h);
    }
    if (dropblock && !dropblock_size_rel && !dropblock_size_abs) {
        printf(" [dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) are set, will be used: dropblock_size_abs = 7 \n");
        dropblock_size_abs = 7;
    }
    if (dropblock_size_rel && dropblock_size_abs) {
        printf(" [dropout] - Both parameters are set, only the parameter will be used: dropblock_size_abs = %d \n", dropblock_size_abs);
        dropblock_size_rel = 0;
    }
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability, dropblock, dropblock_size_rel, dropblock_size_abs, params.w, params.h, params.c);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c, params.train);
    return l;
}

layer parse_shortcut(list *options, size_params params, network net)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    char *weights_type_str = option_find_str_quiet(options, "weights_type", "none");
    WEIGHTS_TYPE_T weights_type = NO_WEIGHTS;
    if(strcmp(weights_type_str, "per_feature") == 0 || strcmp(weights_type_str, "per_layer") == 0) weights_type = PER_FEATURE;
    else if (strcmp(weights_type_str, "per_channel") == 0) weights_type = PER_CHANNEL;
    else if (strcmp(weights_type_str, "none") != 0) {
        printf("Error: Incorrect weights_type = %s \n Use one of: none, per_feature, per_channel \n", weights_type_str);
        getchar();
        exit(0);
    }

    char *weights_normalization_str = option_find_str_quiet(options, "weights_normalization", "none");
    WEIGHTS_NORMALIZATION_T weights_normalization = NO_NORMALIZATION;
    if (strcmp(weights_normalization_str, "relu") == 0 || strcmp(weights_normalization_str, "avg_relu") == 0) weights_normalization = RELU_NORMALIZATION;
    else if (strcmp(weights_normalization_str, "softmax") == 0) weights_normalization = SOFTMAX_NORMALIZATION;
    else if (strcmp(weights_type_str, "none") != 0) {
        printf("Error: Incorrect weights_normalization = %s \n Use one of: none, relu, softmax \n", weights_normalization_str);
        getchar();
        exit(0);
    }

    char *l = option_find(options, "from");
    int len = strlen(l);
    if (!l) error("Route Layer must specify input layers: from = ...", DARKNET_LOC);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }

    int* layers = (int*)calloc(n, sizeof(int));
    int* sizes = (int*)calloc(n, sizeof(int));
    float **layers_output = (float **)calloc(n, sizeof(float *));
    float **layers_delta = (float **)calloc(n, sizeof(float *));
    float **layers_output_gpu = (float **)calloc(n, sizeof(float *));
    float **layers_delta_gpu = (float **)calloc(n, sizeof(float *));

    for (i = 0; i < n; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = params.net.layers[index].outputs;
        layers_output[i] = params.net.layers[index].output;
        layers_delta[i] = params.net.layers[index].delta;
    }

#ifdef GPU
    for (i = 0; i < n; ++i) {
        layers_output_gpu[i] = params.net.layers[layers[i]].output_gpu;
        layers_delta_gpu[i] = params.net.layers[layers[i]].delta_gpu;
    }
#endif// GPU

    layer s = make_shortcut_layer(params.batch, n, layers, sizes, params.w, params.h, params.c, layers_output, layers_delta,
        layers_output_gpu, layers_delta_gpu, weights_type, weights_normalization, activation, params.train);

    free(layers_output_gpu);
    free(layers_delta_gpu);

    for (i = 0; i < n; ++i) {
        int index = layers[i];
        assert(params.w == net.layers[index].out_w && params.h == net.layers[index].out_h);

        if (params.w != net.layers[index].out_w || params.h != net.layers[index].out_h || params.c != net.layers[index].out_c)
            fprintf(stderr, " (%4d x%4d x%4d) + (%4d x%4d x%4d) \n",
                params.w, params.h, params.c, net.layers[index].out_w, net.layers[index].out_h, params.net.layers[index].out_c);
    }

    return s;
}


layer parse_scale_channels(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;
    int scale_wh = option_find_int_quiet(options, "scale_wh", 0);

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_scale_channels_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c, scale_wh);

    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    if (activation == SWISH || activation == MISH) {
        printf(" [scale_channels] layer doesn't support SWISH or MISH activations \n");
    }
    return s;
}

layer parse_sam(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_sam_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    if (activation == SWISH || activation == MISH) {
        printf(" [sam] layer doesn't support SWISH or MISH activations \n");
    }
    return s;
}

layer parse_implicit(list *options, size_params params, network net)
{
    float mean_init = option_find_float(options, "mean", 0.0);
    float std_init = option_find_float(options, "std", 0.2);
    int filters = option_find_int(options, "filters", 128);
    int atoms = option_find_int_quiet(options, "atoms", 1);

    layer s = make_implicit_layer(params.batch, params.index, mean_init, std_init, filters, atoms);

    return s;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.out_h = params.h;
    l.out_w = params.w;
    l.out_c = params.c;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network net)
{

    int stride = option_find_int(options, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

route_layer parse_route(list *options, size_params params)
{
    char *l = option_find(options, "layers");
    if(!l) error("Route Layer must specify input layers", DARKNET_LOC);
    int len = strlen(l);
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int* layers = (int*)xcalloc(n, sizeof(int));
    int* sizes = (int*)xcalloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = params.net.layers[index].outputs;
    }
    int batch = params.batch;

    int groups = option_find_int_quiet(options, "groups", 1);
    int group_id = option_find_int_quiet(options, "group_id", 0);

    route_layer layer = make_route_layer(batch, n, layers, sizes, groups, group_id);

    convolutional_layer first = params.net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = params.net.layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            fprintf(stderr, " The width and height of the input layers are different. \n");
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }
    layer.out_c = layer.out_c / layer.groups;

    layer.w = first.w;
    layer.h = first.h;
    layer.c = layer.out_c;

    layer.stream = option_find_int_quiet(options, "stream", -1);
    layer.wait_stream_id = option_find_int_quiet(options, "wait_stream", -1);

    if (n > 3) fprintf(stderr, " \t    ");
    else if (n > 1) fprintf(stderr, " \t            ");
    else fprintf(stderr, " \t\t            ");

    fprintf(stderr, "           ");
    if (layer.groups > 1) fprintf(stderr, "%d/%d", layer.group_id, layer.groups);
    else fprintf(stderr, "   ");
    fprintf(stderr, " -> %4d x%4d x%4d \n", layer.out_w, layer.out_h, layer.out_c);

    return layer;
}



void set_train_only_bn(network net)
{
    int train_only_bn = 0;
    int i;
    for (i = net.n - 1; i >= 0; --i) {
        if (net.layers[i].train_only_bn) train_only_bn = net.layers[i].train_only_bn;  // set l.train_only_bn for all previous layers
        if (train_only_bn) {
            net.layers[i].train_only_bn = train_only_bn;

            if (net.layers[i].type == CONV_LSTM) {
                net.layers[i].wf->train_only_bn = train_only_bn;
                net.layers[i].wi->train_only_bn = train_only_bn;
                net.layers[i].wg->train_only_bn = train_only_bn;
                net.layers[i].wo->train_only_bn = train_only_bn;
                net.layers[i].uf->train_only_bn = train_only_bn;
                net.layers[i].ui->train_only_bn = train_only_bn;
                net.layers[i].ug->train_only_bn = train_only_bn;
                net.layers[i].uo->train_only_bn = train_only_bn;
                if (net.layers[i].peephole) {
                    net.layers[i].vf->train_only_bn = train_only_bn;
                    net.layers[i].vi->train_only_bn = train_only_bn;
                    net.layers[i].vo->train_only_bn = train_only_bn;
                }
            }
            else if (net.layers[i].type == CRNN) {
                net.layers[i].input_layer->train_only_bn = train_only_bn;
                net.layers[i].self_layer->train_only_bn = train_only_bn;
                net.layers[i].output_layer->train_only_bn = train_only_bn;
            }
        }
    }
}

network parse_network_cfg(char *filename)
{
    return parse_network_cfg_custom(filename, /* batch = */ 0, /* timesteps = */ 0);
}





void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int size = (l.c/l.groups)*l.size*l.size;
    binarize_weights(l.weights, l.n, size, l.binary_weights);
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_shortcut_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_shortcut_layer(l);
        printf("\n pull_shortcut_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_implicit_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_implicit_layer(l);
        //printf("\n pull_implicit_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    //for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_convolutional_weights_ema(layer l, FILE *fp)
{
    if (l.binary) {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if (gpu_index >= 0) {
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases_ema, sizeof(float), l.n, fp);
    if (l.batch_normalize) {
        fwrite(l.scales_ema, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights_ema, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.c, fp);
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network net, char *filename, int cutoff, int save_ema)
{
#ifdef GPU
    if(net.gpu_index >= 0){
        cuda_set_device(net.gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = MAJOR_VERSION;
    int minor = MINOR_VERSION;
    int revision = PATCH_VERSION;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    (*net.seen) = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
    fwrite(net.seen, sizeof(uint64_t), 1, fp);

    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.share_layer == NULL) {
            if (save_ema) {
                save_convolutional_weights_ema(l, fp);
            }
            else {
                save_convolutional_weights(l, fp);
            }
        } if (l.type == SHORTCUT && l.nweights > 0) {
            save_shortcut_weights(l, fp);
        } if (l.type == IMPLICIT) {
            save_implicit_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if(l.type == GRU){
            save_connected_weights(*(l.input_z_layer), fp);
            save_connected_weights(*(l.input_r_layer), fp);
            save_connected_weights(*(l.input_h_layer), fp);
            save_connected_weights(*(l.state_z_layer), fp);
            save_connected_weights(*(l.state_r_layer), fp);
            save_connected_weights(*(l.state_h_layer), fp);
        } if(l.type == LSTM){
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.ug), fp);
            save_connected_weights(*(l.uo), fp);
        } if (l.type == CONV_LSTM) {
            if (l.peephole) {
                save_convolutional_weights(*(l.vf), fp);
                save_convolutional_weights(*(l.vi), fp);
                save_convolutional_weights(*(l.vo), fp);
            }
            save_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                save_convolutional_weights(*(l.wi), fp);
                save_convolutional_weights(*(l.wg), fp);
                save_convolutional_weights(*(l.wo), fp);
            }
            save_convolutional_weights(*(l.uf), fp);
            save_convolutional_weights(*(l.ui), fp);
            save_convolutional_weights(*(l.ug), fp);
            save_convolutional_weights(*(l.uo), fp);
        } if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
        fflush(fp);
    }
    fclose(fp);
}
void save_weights(network net, char *filename)
{
    save_weights_upto(net, filename, net.n, 0);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.c, fp);
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = (l.c / l.groups)*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.biases, sizeof(float), l.n, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", l.index);
    //fread(l.weights, sizeof(float), num, fp); // as in connected layer
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes = fread(l.scales, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", l.index);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //if(l.adam){
    //    fread(l.m, sizeof(float), num, fp);
    //    fread(l.v, sizeof(float), num, fp);
    //}
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_shortcut_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_shortcut_layer(l);
    }
#endif
}

void load_implicit_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_implicit_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        printf("\n seen 64");
        uint64_t iseen = 0;
        fread(&iseen, sizeof(uint64_t), 1, fp);
        *net->seen = iseen;
    }
    else {
        printf("\n seen 32");
        uint32_t iseen = 0;
        fread(&iseen, sizeof(uint32_t), 1, fp);
        *net->seen = iseen;
    }
    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
            load_convolutional_weights(l, fp);
        }
        if (l.type == SHORTCUT && l.nweights > 0) {
            load_shortcut_weights(l, fp);
        }
        if (l.type == IMPLICIT) {
            load_implicit_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == GRU){
            load_connected_weights(*(l.input_z_layer), fp, transpose);
            load_connected_weights(*(l.input_r_layer), fp, transpose);
            load_connected_weights(*(l.input_h_layer), fp, transpose);
            load_connected_weights(*(l.state_z_layer), fp, transpose);
            load_connected_weights(*(l.state_r_layer), fp, transpose);
            load_connected_weights(*(l.state_h_layer), fp, transpose);
        }
        if(l.type == LSTM){
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
        }
        if (l.type == CONV_LSTM) {
            if (l.peephole) {
                load_convolutional_weights(*(l.vf), fp);
                load_convolutional_weights(*(l.vi), fp);
                load_convolutional_weights(*(l.vo), fp);
            }
            load_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                load_convolutional_weights(*(l.wi), fp);
                load_convolutional_weights(*(l.wg), fp);
                load_convolutional_weights(*(l.wo), fp);
            }
            load_convolutional_weights(*(l.uf), fp);
            load_convolutional_weights(*(l.ui), fp);
            load_convolutional_weights(*(l.ug), fp);
            load_convolutional_weights(*(l.uo), fp);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
        if (feof(fp)) break;
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}
