// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


namespace Darknet_ng
{
	/** The @p %Network objects contains everything we need to know about the neural network.
	 * These objects can be instantiated on the stack.
	 *
	 * @since 2022-11-13
	 */
	class Network final
	{
		public:

			/// Destructor.
			~Network();

			/// Constructor.  You'll need to call @ref load().
			Network();

			/// Constructor.  This automatically calls @ref load().
			Network(const std::filesystem::path & cfg_filename);

			/// Reset all of the network configuration so the object can be re-used.  You'll need to call @ref load() after @ref clear().
			Network & clear();

			/// @{ Determines if a neural network has been loaded.
			bool empty() const;
			bool loaded() const { return not empty(); }
			/// @}

			/** Load the given network.  This is automatically called by the constructor when a filename has been provided,
			 * or it can be manually called with a specific filename to trigger the network to load.
			 */
			Network & load(const std::filesystem::path & cfg_filename);

			/// @todo
			Network & make_network(const Config & cfg);

			/// @todo
			Network & parse_layers(const Config & cfg);

			/// @{ Parse the given section from the configuration.  This is automatically called by @ref load().
			Network & parse_net				(const Section & section);
			Network & parse_convolutional	(const Section & section, const size_t layer_index);
			/// @}

			/** All of the fields in this structure must be POD ("plain old data") since they're reset in bulk via the use of
			 * @p std::memset() in @ref Network::clear().  Anything more complex than POD such as vectors and maps are defined
			 * outside of this structure and need to be manually handled in @ref Network::clear().
			 *
			 * @since 2022-11-18
			 */
			struct Settings final
			{
				int gpu_index;						///< set to -1 for no GPU or >= 0 to use the given GPU
				int max_batches;					///< [net][max_batches]
				int batch;							///< [net][batch]
				float learning_rate;				///< [net][learning_rate]
				float learning_rate_min;			///< [net][learning_rate_min]
				int batches_per_cycle;				///< [net][sgdr_cycle]
				int batches_cycle_mult;				///< [net][sgdr_mult]
				float momentum;						///< [net][momentum]
				float decay;						///< [net][decay]
				int subdivisions;					///< [net][subdivisions]
				int time_steps;						///< [net][time_steps]
				int track;							///< [net][track]
				int augment_speed;					///< [net][augment_speed]
				int init_sequential_subdivisions;	///< [net][sequential_subdivisions]
				int sequential_subdivisions;		///< [net][sequential_subdivisions]
				int try_fix_nan;					///< [net][try_fix_nan]
				int weights_reject_freq;			///< [net][weights_reject_freq]
				int equidistant_point;				///< [net][equidistant_point]
				float badlabels_rejection_percentage; ///< [net][badlabels_rejection_percentage]
				float num_sigmas_reject_badlabels;	///< [net][num_sigmas_reject_badlabels]
				float ema_alpha;					///< [net][ema_alpha]

				/// @{ @todo why were these next 7 items pointers?
				float badlabels_reject_threshold;
				float delta_rolling_max;
				float delta_rolling_avg;
				float delta_rolling_std;
				uint64_t seen;
				int cur_iteration;
				bool cuda_graph_ready;
				/// @}

				bool use_cuda_graph;				///< [net][use_cuda_graph]
				float loss_scale;					///< [net][loss_scale]
				int dynamic_minibatch;				///< [net][dynamic_minibatch]
				int optimized_memory;				///< [net][optimized_memory]
				size_t workspace_size_limit;		///< [net][workspace_size_limit_MB]

				bool adam;							///< [net][adam]
				float B1;							///< [net][B1]
				float B2;							///< [net][B2]
				float eps;							///< [net][eps]

				int h;								///< height
				int w;								///< width
				int c;								///< channels

				int inputs;							///< [net][inputs]
				int max_crop;						///< [net][max_crop]
				int min_crop;						///< [net][min_crop]
				bool flip;							///< horizontal flip 50% probability augmentation for classifier training (default = 1)
				int blur;							///< [net][blur]
				int gaussian_noise;					///< [net][gaussian_noise]
				int mixup;							///< [net][mixup]
				int letter_box;						///< [net][letter_box]
				int mosaic_bound;					///< [net][mosaic_bound]
				int contrastive;					///< [net][contrastive]
				int contrastive_jit_flip;			///< [net][contrastive_jit_flip]
				int contrastive_color;				///< [net][contrastive_color]
				int unsupervised;					///< [net][unsupervised]
				float label_smooth_eps;				///< [net][label_smooth_eps]
				int resize_step;					///< [net][resize_step]
				int attention;						///< [net][attention]
				float adversarial_lr;				///< [net][adversarial_lr]
				float max_chart_loss;				///< [net][max_chart_loss]
				float angle;						///< [net][angle]
				float aspect;						///< [net][aspect]
				float saturation;					///< [net][saturation]
				float exposure;						///< [net][exposure]
				float hue;							///< [net][hue]
				float power;						///< [net][power]

				ELearningRatePolicy policy;			///< [net][policy]

				int burn_in;						///< [net][burn_in]
				int step;							///< [net][step]
				float scale;						///< [net][scale]

				int num_steps;						///< number of entries in @ref steps @todo this can be removed since "steps" is now much easier to manage
				float gamma;						///< [net][gamma]

				bool train; // was int, converting to a bool

				/* WARNING: Only POD (plain-old-data) can go in this structure!  No objects, meaning no std::strings.
				 * Limit yourself to enums, bools, ints, and floats.  See comment above explaining why.
				 */
			};
			/// Most (but not quite all) of the configuration for a neural network.  See the comments for @ref Network::Settings.
			Settings settings;

			VI steps;		///< [net][steps]
			VF scales;		///< [net][scales]
			VF seq_scales;	///< [net][seq_scales]

			/// Network layers.  @see @ref load()
			Layers layers;


#ifdef WORK_IN_PROGRESS /// @todo
			int n;	// the number of layers in the network (sections - 1, since [net] doesn't count)
			int *t;
			float epoch;
			Layer *layers;
			float *output;
			int benchmark_layers;
			int *total_bbox;
			int *rewritten_bbox;

			float learning_rate_max;
			int num_boxes;
			int train_images_num;
			int cudnn_half;

			int outputs;
			int truths;
			int notruth;
			float max_ratio;
			float min_ratio;
			int center;
			int adversarial;
			int random;
			int current_subdivision;

			Tree *hierarchy;

			float *input;
			float *truth;
			float *delta;
			float *workspace;
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


			float *global_delta_gpu;
			float *state_delta_gpu;
			size_t max_delta_gpu_size;
			//#endif  // GPU
#endif
	};

	int convolutional_out_width(const Layer & layer);
	int convolutional_out_height(const Layer & layer);
	size_t get_convolutional_workspace_size(const Layer & layer);
}
