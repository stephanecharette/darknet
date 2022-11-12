// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "enums.hpp"


namespace Darknet_ng
{
#ifdef STEPHANE
	struct Section // was: section
	{
		char *type;
		list *options;
		MStr options;
	};
#endif

	struct ContrastiveParams // was: contrastive_params
	{
		float	sim;
		float	exp_sim;
		float	P;
		int		i;
		int		j;
		int		time_step_i;
		int		time_step_j;
	};

	struct Tree // was: tree
	{
		int *leaf;
		int n;
		int *parent;
		int *child;
		int *group;
		char **name;

		int groups;
		int *group_size;
		int *group_offset;
	};

	struct UpdateArgs
	{
		int		batch;
		float	learning_rate;
		float	momentum;
		float	decay;
		int		adam;
		float	B1;
		float	B2;
		float	eps;
		int		t;
	};

	struct Layer
	{
		ELayerType		type;
		EActivation		activation;
		EActivation		lstm_activation;
		ECostType		cost_type;

		void(*forward)   (struct layer, struct network_state);
		void(*backward)  (struct layer, struct network_state);
		void(*update)    (struct layer, int, float, float, float);
		void(*forward_gpu)   (struct layer, struct network_state);
		void(*backward_gpu)  (struct layer, struct network_state);
		void(*update_gpu)    (struct layer, int, float, float, float, float);
		Layer *share_layer;
		int train;
		int avgpool;
		int batch_normalize;
		int shortcut;
		int batch;
		int dynamic_minibatch;
		int forced;
		int flipped;
		int inputs;
		int outputs;
		float mean_alpha;
		int nweights;
		int nbiases;
		int extra;
		int truths;
		int h;
		int w;
		int c;
		int out_h;
		int out_w;
		int out_c;
		int n;
		int max_boxes;
		int truth_size;
		int groups;
		int group_id;
		int size;
		int side;
		int stride;
		int stride_x;
		int stride_y;
		int dilation;
		int antialiasing;
		int maxpool_depth;
		int maxpool_zero_nonmax;
		int out_channels;
		float reverse;
		int coordconv;
		int flatten;
		int spatial;
		int pad;
		int sqrt;
		int flip;
		int index;
		int scale_wh;
		int binary;
		int xnor;
		int peephole;
		int use_bin_output;
		int keep_delta_gpu;
		int optimized_memory;
		int steps;
		int history_size;
		int bottleneck;
		float time_normalizer;
		int state_constrain;
		int hidden;
		int truth;
		float smooth;
		float dot;
		int deform;
		int grad_centr;
		int sway;
		int rotate;
		int stretch;
		int stretch_sway;
		float angle;
		float jitter;
		float resize;
		float saturation;
		float exposure;
		float shift;
		float ratio;
		float learning_rate_scale;
		float clip;
		int focal_loss;
		float *classes_multipliers;
		float label_smooth_eps;
		int noloss;
		int softmax;
		int classes;
		int detection;
		int embedding_layer_id;
		float *embedding_output;
		int embedding_size;
		float sim_thresh;
		int track_history_size;
		int dets_for_track;
		int dets_for_show;
		float track_ciou_norm;
		int coords;
		int background;
		int rescore;
		int objectness;
		int does_cost;
		int joint;
		int noadjust;
		int reorg;
		int log;
		int tanh;
		int *mask;
		int total;
		float bflops;

		int adam;
		float B1;
		float B2;
		float eps;

		int t;

		float alpha;
		float beta;
		float kappa;

		float coord_scale;
		float object_scale;
		float noobject_scale;
		float mask_scale;
		float class_scale;
		int bias_match;
		float random;
		float ignore_thresh;
		float truth_thresh;
		float iou_thresh;
		float thresh;
		float focus;
		int classfix;
		int absolute;
		int assisted_excitation;

		int onlyforward;
		int stopbackward;
		int train_only_bn;
		int dont_update;
		int burnin_update;
		int dontload;
		int dontsave;
		int dontloadscales;
		int numload;

		float temperature;
		float probability;
		float dropblock_size_rel;
		int dropblock_size_abs;
		int dropblock;
		float scale;

		int receptive_w;
		int receptive_h;
		int receptive_w_scale;
		int receptive_h_scale;

		char  * cweights;
		int   * indexes;
		int   * input_layers;
		int   * input_sizes;
		float **layers_output;
		float **layers_delta;
		EWeightsType weights_type; // WEIGHTS_TYPE_T
		EWeightsNormalization weights_normalization; // WEIGHTS_NORMALIZATION_T
		int   * map;
		int   * counts;
		float ** sums;
		float * rand;
		float * cost;
		int *labels;
		int *class_ids;
		int contrastive_neg_max;
		float *cos_sim;
		float *exp_cos_sim;
		float *p_constrastive;
		ContrastiveParams *contrast_p_gpu;
		float * state;
		float * prev_state;
		float * forgot_state;
		float * forgot_delta;
		float * state_delta;
		float * combine_cpu;
		float * combine_delta_cpu;

		float *concat;
		float *concat_delta;

		float *binary_weights;

		float *biases;
		float *bias_updates;

		float *scales;
		float *scale_updates;

		float *weights_ema;
		float *biases_ema;
		float *scales_ema;

		float *weights;
		float *weight_updates;

		float scale_x_y;
		int objectness_smooth;
		int new_coords;
		int show_details;
		float max_delta;
		float uc_normalizer;
		float iou_normalizer;
		float obj_normalizer;
		float cls_normalizer;
		float delta_normalizer;
		EIOULoss iou_loss;
		EIOULoss iou_thresh_kind;
		ENMSKind nms_kind;
		float beta_nms;
		EYOLOPoint yolo_point;

		char *align_bit_weights_gpu;
		float *mean_arr_gpu;
		float *align_workspace_gpu;
		float *transposed_align_workspace_gpu;
		int align_workspace_size;

		char *align_bit_weights;
		float *mean_arr;
		int align_bit_weights_size;
		int lda_align;
		int new_lda;
		int bit_align;

		float *col_image;
		float * delta;
		float * output;
		float * activation_input;
		int delta_pinned;
		int output_pinned;
		float * loss;
		float * squared;
		float * norms;

		float * spatial_mean;
		float * mean;
		float * variance;

		float * mean_delta;
		float * variance_delta;

		float * rolling_mean;
		float * rolling_variance;

		float * x;
		float * x_norm;

		float * m;
		float * v;

		float * bias_m;
		float * bias_v;
		float * scale_m;
		float * scale_v;

		float *z_cpu;
		float *r_cpu;
		float *h_cpu;
		float *stored_h_cpu;
		float * prev_state_cpu;

		float *temp_cpu;
		float *temp2_cpu;
		float *temp3_cpu;

		float *dh_cpu;
		float *hh_cpu;
		float *prev_cell_cpu;
		float *cell_cpu;
		float *f_cpu;
		float *i_cpu;
		float *g_cpu;
		float *o_cpu;
		float *c_cpu;
		float *stored_c_cpu;
		float *dc_cpu;

		float *binary_input;
		uint32_t *bin_re_packed_input;
		char *t_bit_input;

		struct layer *input_layer;
		struct layer *self_layer;
		struct layer *output_layer;

		struct layer *reset_layer;
		struct layer *update_layer;
		struct layer *state_layer;

		struct layer *input_gate_layer;
		struct layer *state_gate_layer;
		struct layer *input_save_layer;
		struct layer *state_save_layer;
		struct layer *input_state_layer;
		struct layer *state_state_layer;

		struct layer *input_z_layer;
		struct layer *state_z_layer;

		struct layer *input_r_layer;
		struct layer *state_r_layer;

		struct layer *input_h_layer;
		struct layer *state_h_layer;

		struct layer *wz;
		struct layer *uz;
		struct layer *wr;
		struct layer *ur;
		struct layer *wh;
		struct layer *uh;
		struct layer *uo;
		struct layer *wo;
		struct layer *vo;
		struct layer *uf;
		struct layer *wf;
		struct layer *vf;
		struct layer *ui;
		struct layer *wi;
		struct layer *vi;
		struct layer *ug;
		struct layer *wg;

		Tree *softmax_tree;

		size_t workspace_size;

		//#ifdef GPU
		int *indexes_gpu;

		int stream;
		int wait_stream_id;

		float *z_gpu;
		float *r_gpu;
		float *h_gpu;
		float *stored_h_gpu;
		float *bottelneck_hi_gpu;
		float *bottelneck_delta_gpu;

		float *temp_gpu;
		float *temp2_gpu;
		float *temp3_gpu;

		float *dh_gpu;
		float *hh_gpu;
		float *prev_cell_gpu;
		float *prev_state_gpu;
		float *last_prev_state_gpu;
		float *last_prev_cell_gpu;
		float *cell_gpu;
		float *f_gpu;
		float *i_gpu;
		float *g_gpu;
		float *o_gpu;
		float *c_gpu;
		float *stored_c_gpu;
		float *dc_gpu;

		// adam
		float *m_gpu;
		float *v_gpu;
		float *bias_m_gpu;
		float *scale_m_gpu;
		float *bias_v_gpu;
		float *scale_v_gpu;

		float * combine_gpu;
		float * combine_delta_gpu;

		float * forgot_state_gpu;
		float * forgot_delta_gpu;
		float * state_gpu;
		float * state_delta_gpu;
		float * gate_gpu;
		float * gate_delta_gpu;
		float * save_gpu;
		float * save_delta_gpu;
		float * concat_gpu;
		float * concat_delta_gpu;

		float *binary_input_gpu;
		float *binary_weights_gpu;
		float *bin_conv_shortcut_in_gpu;
		float *bin_conv_shortcut_out_gpu;

		float * mean_gpu;
		float * variance_gpu;
		float * m_cbn_avg_gpu;
		float * v_cbn_avg_gpu;

		float * rolling_mean_gpu;
		float * rolling_variance_gpu;

		float * variance_delta_gpu;
		float * mean_delta_gpu;

		float * col_image_gpu;

		float * x_gpu;
		float * x_norm_gpu;
		float * weights_gpu;
		float * weight_updates_gpu;
		float * weight_deform_gpu;
		float * weight_change_gpu;

		float * weights_gpu16;
		float * weight_updates_gpu16;

		float * biases_gpu;
		float * bias_updates_gpu;
		float * bias_change_gpu;

		float * scales_gpu;
		float * scale_updates_gpu;
		float * scale_change_gpu;

		float * input_antialiasing_gpu;
		float * output_gpu;
		float * output_avg_gpu;
		float * activation_input_gpu;
		float * loss_gpu;
		float * delta_gpu;
		float * cos_sim_gpu;
		float * rand_gpu;
		float * drop_blocks_scale;
		float * drop_blocks_scale_gpu;
		float * squared_gpu;
		float * norms_gpu;

		float *gt_gpu;
		float *a_avg_gpu;

		int *input_sizes_gpu;
		float **layers_output_gpu;
		float **layers_delta_gpu;
		#ifdef CUDNN
		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
		cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
		cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
		cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
		cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
		cudnnFilterDescriptor_t weightDesc, weightDesc16;
		cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
		cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
		cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
		cudnnPoolingDescriptor_t poolingDesc;
		#else   // CUDNN
		void* srcTensorDesc, *dstTensorDesc;
		void* srcTensorDesc16, *dstTensorDesc16;
		void* dsrcTensorDesc, *ddstTensorDesc;
		void* dsrcTensorDesc16, *ddstTensorDesc16;
		void* normTensorDesc, *normDstTensorDesc, *normDstTensorDescF16;
		void* weightDesc, *weightDesc16;
		void* dweightDesc, *dweightDesc16;
		void* convDesc;
		UNUSED_ENUM_TYPE fw_algo, fw_algo16;
		UNUSED_ENUM_TYPE bd_algo, bd_algo16;
		UNUSED_ENUM_TYPE bf_algo, bf_algo16;
		void* poolingDesc;
		#endif  // CUDNN
		//#endif  // GPU
	};

	struct Network /// was: network
	{
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
		Layer *layers;
		float *output;
		ELearningRatePolicy policy;
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
		Tree *hierarchy;

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
	};

	struct NetworkState /// was: network_state
	{
		float *truth;
		float *input;
		float *delta;
		float *workspace;
		int train;
		int index;
		Network net;
	};

	struct Box /// was: box
	{
		float x;
		float y;
		float w;
		float h;
	};

	struct Detection /// was: detection
	{
		Box bbox;
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
	};
}
