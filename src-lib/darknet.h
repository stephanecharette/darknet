#ifndef DARKNET_API
#define DARKNET_API

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif

#define SECRET_NUM -1234

typedef enum { UNUSED_DEF_VAL } UNUSED_ENUM_TYPE;

#ifdef GPU

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif  // CUDNN
#endif  // GPU

#ifdef __cplusplus
extern "C" {
#endif

struct network;
typedef struct network network;

struct network_state;
typedef struct network_state network_state;

struct layer;
typedef struct layer layer;

struct image;
typedef struct image image;

struct detection;
typedef struct detection detection;

struct load_args;
typedef struct load_args load_args;

struct data;
typedef struct data data;

struct metadata;
typedef struct metadata metadata;

struct tree;
typedef struct tree tree;

extern int gpu_index;

// option_list.h
typedef struct metadata {
    int classes;
    char **names;
} metadata;


// tree.h
typedef struct tree {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

// blas.h
typedef struct contrastive_params {
    float sim;
    float exp_sim;
    float P;
    int i, j;
    int time_step_i, time_step_j;
} contrastive_params;


// layers moved to layers.hpp

// network.h
typedef struct network {
    int n;
    int batch;
    uint64_t *seen;
    float *badlabels_reject_threshold;
    float *delta_rolling_max;
    float *delta_rolling_avg;
    float *delta_rolling_std;
    int weights_reject_freq;
    int equidistant_point;
    float badlabels_rejection_percentage;
    float num_sigmas_reject_badlabels;
    float ema_alpha;
    int *cur_iteration;
    float loss_scale;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;
    int benchmark_layers;
    int *total_bbox;
    int *rewritten_bbox;

    float learning_rate;
    float learning_rate_min;
    float learning_rate_max;
    int batches_per_cycle;
    int batches_cycle_mult;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    int num_boxes;
    int train_images_num;
    float *seq_scales;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    int gaussian_noise;
    int blur;
    int mixup;
    float label_smooth_eps;
    int resize_step;
    int attention;
    int adversarial;
    float adversarial_lr;
    float max_chart_loss;
    int letter_box;
    int mosaic_bound;
    int contrastive;
    int contrastive_jit_flip;
    int contrastive_color;
    int unsupervised;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;
    int track;
    int augment_speed;
    int sequential_subdivisions;
    int init_sequential_subdivisions;
    int current_subdivision;
    int try_fix_nan;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

//#ifdef GPU
    //float *input_gpu;
    //float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;

    float *input_state_gpu;
    float *input_pinned_cpu;
    int input_pinned_cpu_flag;

    float **input_gpu;
    float **truth_gpu;
    float **input16_gpu;
    float **output16_gpu;
    size_t *max_input16_size;
    size_t *max_output16_size;
    int wait_stream;

    void *cuda_graph;
    void *cuda_graph_exec;
    int use_cuda_graph;
    int *cuda_graph_ready;

    float *global_delta_gpu;
    float *state_delta_gpu;
    size_t max_delta_gpu_size;
//#endif  // GPU
    int optimized_memory;
    int dynamic_minibatch;
    size_t workspace_size_limit;
} network;

// network.h
typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;

//typedef struct {
//    int w;
//    int h;
//    float scale;
//    float rad;
//    float dx;
//    float dy;
//    float aspect;
//} augment_args;

// image.h
typedef struct image {
    int w;
    int h;
    int c;
    float *data;
} image;

//typedef struct {
//    int w;
//    int h;
//    int c;
//    float *data;
//} image;

// box.h
typedef struct box {
    float x, y, w, h;
} box;

// box.h
typedef struct boxabs {
    float left, right, top, bot;
} boxabs;

// box.h
typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;


// box.h
typedef struct detection{
    box bbox;
    int classes;
    int best_class_idx;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
    float *embeddings;  // embeddings for tracking
    int embedding_size;
    float sim;
    int track_id;
} detection;

// network.c -batch inference
typedef struct det_num_pair {
    int num;
    detection *dets;
} det_num_pair, *pdet_num_pair;

// matrix.h
typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

// data.h
typedef struct data {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

// data.h
typedef struct load_args {
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int c; // color depth
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int truth_size;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    int mini_batch;
    int track;
    int augment_speed;
    int letter_box;
    int mosaic_bound;
    int show_imgs;
    int dontuse_opencv;
    int contrastive;
    int contrastive_jit_flip;
    int contrastive_color;
    float jitter;
    float resize;
    int flip;
    int gaussian_noise;
    int blur;
    int mixup;
    float label_smooth_eps;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

// data.h
typedef struct box_label {
    int id;
    int track_id;
    float x, y, w, h;
    float left, right, top, bottom;
} box_label;

// list.h
//typedef struct node {
//    void *val;
//    struct node *next;
//    struct node *prev;
//} node;

// list.h
//typedef struct list {
//    int size;
//    node *front;
//    node *back;
//} list;
// -----------------------------------------------------


// parser.c
LIB_API network *load_network(char *cfg, char *weights, int clear);
LIB_API network *load_network_custom(char *cfg, char *weights, int clear, int batch);
LIB_API void free_network(network net);
LIB_API void free_network_ptr(network* net);

// network.c
LIB_API load_args get_base_args(network *net);

// box.h
LIB_API void do_nms_sort(detection *dets, int total, int classes, float thresh);
LIB_API void do_nms_obj(detection *dets, int total, int classes, float thresh);
LIB_API void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

// network.h
LIB_API float *network_predict(network net, float *input);
LIB_API float *network_predict_ptr(network *net, float *input);
LIB_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
LIB_API det_num_pair* network_predict_batch(network *net, image im, int batch_size, int w, int h, float thresh, float hier, int *map, int relative, int letter);
LIB_API void free_detections(detection *dets, int n);
LIB_API void free_batch_detections(det_num_pair *det_num_pairs, int n);
LIB_API void fuse_conv_batchnorm(network net);
LIB_API void calculate_binary_weights(network net);
LIB_API char *detection_to_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, char *filename);

LIB_API layer* get_network_layer(network* net, int i);
//LIB_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
LIB_API detection *make_network_boxes(network *net, float thresh, int *num);
LIB_API void reset_rnn(network *net);
LIB_API float *network_predict_image(network *net, image im);
LIB_API float *network_predict_image_letterbox(network *net, image im);
LIB_API float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net);
LIB_API void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path);
LIB_API void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);
LIB_API int network_width(network *net);
LIB_API int network_height(network *net);
LIB_API void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm);

// image.h
LIB_API void make_image_red(image im);
LIB_API image make_attention_image(int img_size, float *original_delta_cpu, float *original_input_cpu, int w, int h, int c, float alpha);
LIB_API image resize_image(image im, int w, int h);
LIB_API void quantize_image(image im);
LIB_API void copy_image_from_bytes(image im, char *pdata);
LIB_API image letterbox_image(image im, int w, int h);
LIB_API void rgbgr_image(image im);
LIB_API image make_image(int w, int h, int c);
LIB_API image load_image_color(char *filename, int w, int h);
LIB_API void free_image(image m);
LIB_API image crop_image(image im, int dx, int dy, int w, int h);
LIB_API image resize_min(image im, int min);

// layer.h
LIB_API void free_layer_custom(layer l, int keep_cudnn_desc);
LIB_API void free_layer(layer l);

// data.c
LIB_API void free_data(data d);
LIB_API pthread_t load_data(load_args args);
LIB_API void free_load_threads(void *ptr);
LIB_API pthread_t load_data_in_thread(load_args args);
LIB_API void *load_thread(void *ptr);

// dark_cuda.h
LIB_API void cuda_pull_array(float *x_gpu, float *x, size_t n);
LIB_API void cuda_pull_array_async(float *x_gpu, float *x, size_t n);
LIB_API void cuda_set_device(int n);
LIB_API void *cuda_get_context();

// utils.h
LIB_API void free_ptrs(void **ptrs, int n);
LIB_API void top_k(float *a, int n, int k, int *index);

// tree.h
LIB_API tree *read_tree(char *filename);

// option_list.h
LIB_API metadata get_metadata(char *file);


// http_stream.h
LIB_API void delete_json_sender();
LIB_API void send_json_custom(char const* send_buf, int port, int timeout);
LIB_API double get_time_point();
void start_timer();
void stop_timer();
double get_time();
void stop_timer_and_show();
void stop_timer_and_show_name(char *name);
void show_total_time();

LIB_API void set_track_id(detection *new_dets, int new_dets_num, float thresh, float sim_thresh, float track_ciou_norm, int deque_size, int dets_for_track, int dets_for_show);
LIB_API int fill_remaining_id(detection *new_dets, int new_dets_num, int new_track_id, float thresh);


// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // DARKNET_API
