// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include <iostream>
#include <cstring>


Darknet_ng::Network::~Network()
{
	return;
}


Darknet_ng::Network::Network()
{
	clear();

	return;
}


Darknet_ng::Network::Network(const std::filesystem::path & cfg_filename)
{
	load(cfg_filename);

	return;
}


Darknet_ng::Network & Darknet_ng::Network::clear()
{
	std::memset(&settings, '\0', sizeof(Settings));

	// anything more complex than the POD fields in the Settings, or anything
	// that needs to be set to a value other than zero must be handled here

	settings.gpu_index = -1; // do not use the GPU

	steps		.clear();
	scales		.clear();
	seq_scales	.clear();
	layers		.clear();

	return *this;
}


bool Darknet_ng::Network::empty() const
{
	return
		steps		.empty() or
		scales		.empty() or
		seq_scales	.empty() or
		layers		.empty();
}


Darknet_ng::Network & Darknet_ng::Network::load(const std::filesystem::path & cfg_filename)
{
	clear();

	Config cfg(cfg_filename);
	parse_net(cfg);

	return *this;
}


Darknet_ng::Network & Darknet_ng::Network::parse_net(const Config & cfg)
{
	const auto & net = *cfg.sections.begin(); // required to be [net] or [network]

	settings.max_batches					= net.i("max_batches"					, 0);
	settings.batch							= net.i("batch"							, 1);
	settings.learning_rate					= net.f("learning_rate"					, 0.001f);
	settings.learning_rate_min				= net.f("learning_rate_min"				, 0.00001f);
	settings.batches_per_cycle				= net.i("sgdr_cycle"					, settings.max_batches);
	settings.batches_cycle_mult				= net.i("sgdr_mult"						, 2);
	settings.momentum						= net.f("momentum"						, 0.9f);
	settings.decay							= net.f("decay"							, 0.0001f);
	settings.subdivisions					= net.i("subdivisions"					, 1);
	settings.time_steps						= net.i("time_steps"					, 1);
	settings.track							= net.i("track"							, 0);
	settings.augment_speed					= net.i("augment_speed"					, 2);
	settings.init_sequential_subdivisions	= net.i("sequential_subdivisions"		, settings.subdivisions);
	settings.sequential_subdivisions		= settings.init_sequential_subdivisions;

	if (settings.sequential_subdivisions > settings.subdivisions)
	{
		settings.init_sequential_subdivisions	= settings.subdivisions;
		settings.sequential_subdivisions		= settings.subdivisions;
	}

	settings.try_fix_nan						= net.i("try_fix_nan"					, 0);
	settings.batch /= settings.subdivisions; // mini_batch
	const auto mini_batch = settings.batch;
	settings.batch *= settings.time_steps; // mini_batch * time_steps

	settings.weights_reject_freq			= net.i("weights_reject_freq"			, 0);
	settings.equidistant_point				= net.i("equidistant_point"				, 0);
	settings.badlabels_rejection_percentage	= net.f("badlabels_rejection_percentage", 0.0f);
	settings.num_sigmas_reject_badlabels	= net.f("num_sigmas_reject_badlabels"	, 0.0f);
	settings.ema_alpha						= net.f("ema_alpha"						, 0.0f);

	settings.badlabels_reject_threshold		= 0.0f;
	settings.delta_rolling_max				= 0.0f;
	settings.delta_rolling_avg				= 0.0f;
	settings.delta_rolling_std				= 0.0f;
	settings.seen							= 0;
	settings.cur_iteration					= 0;
	settings.cuda_graph_ready				= false;
	settings.use_cuda_graph					= net.b("use_cuda_graph"				, false);
	settings.loss_scale						= net.f("loss_scale"					, 1.0f);
	settings.dynamic_minibatch				= net.i("dynamic_minibatch"				, 0);
	settings.optimized_memory				= net.i("optimized_memory"				, 0);

	/// @todo This is called @p workspace_size_limit_MB but since it is multiplied by 1024*1024, isn't it GiB, not MiB?
	settings.workspace_size_limit			= net.f("workspace_size_limit_MB"		, 1024.0f) * 1024.0f * 1024.0f; // 1 GiB by default

	settings.adam							= net.b("adam"							, false);
	settings.B1								= net.f("B1"							, 0.9f);
	settings.B2								= net.f("B2"							, 0.999f);
	settings.eps							= net.f("eps"							, 0.000001f);

	settings.h								= net.i("height"						,0);
	settings.w								= net.i("width"							,0);
	settings.c								= net.i("channels"						,0);

	settings.inputs							= net.i("inputs"						, settings.h * settings.w * settings.c);
	settings.max_crop						= net.i("max_crop"						, settings.w * 2);
	settings.min_crop						= net.i("min_crop"						, settings.w);
	settings.flip							= net.i("flip"							, true);
	settings.blur							= net.i("blur"							, 0);
	settings.gaussian_noise					= net.i("gaussian_noise"				, 0);
	settings.mixup							= net.i("mixup"							, 0);

	const int cutmix = net.i("cutmix", 0);
	const int mosaic = net.i("mosaic", 0);
	if (mosaic and cutmix)
	{
		settings.mixup = 4;
	}
	else if (mosaic)
	{
		settings.mixup = 3;
	}
	else if (cutmix)
	{
		settings.mixup = 2;
	}

	settings.letter_box						= net.i("letter_box"					, 0);
	settings.mosaic_bound					= net.i("mosaic_bound"					, 0);
	settings.contrastive					= net.i("contrastive"					, 0);
	settings.contrastive_jit_flip			= net.i("contrastive_jit_flip"			, 0);
	settings.contrastive_color				= net.i("contrastive_color"				, 0);
	settings.unsupervised					= net.i("unsupervised"					, 0);

	if (settings.contrastive and mini_batch < 2)
	{
		throw std::runtime_error("mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss");
	}

	settings.label_smooth_eps				= net.f("label_smooth_eps"				, 0.0f);
	settings.resize_step					= net.i("resize_step"					, 32);
	settings.attention						= net.i("attention"						, 0);
	settings.adversarial_lr					= net.f("adversarial_lr"				, 0.0f);
	settings.max_chart_loss					= net.f("max_chart_loss"				, 20.0f);
	settings.angle							= net.f("angle"							, 0.0f);
	settings.aspect							= net.f("aspect"						, 1.0f);
	settings.saturation						= net.f("saturation"					, 1.0f);
	settings.exposure						= net.f("exposure"						, 1.0f);
	settings.hue								= net.f("hue"							, 0.0f);
	settings.power							= net.f("power"							, 4.0f);

	if (not settings.inputs and not (settings.h and settings.w and settings.c))
	{
		throw std::runtime_error("no input parameters supplied");
	}

	settings.policy = learning_rate_policy_from_string(net.s("policy", "constant"));

	settings.burn_in = net.i("burn_in", 0);

	#ifdef GPU
	/// @todo GPU stuff hasn't been touched
	if (net->gpu_index >= 0) {
		char device_name[1024];
		int compute_capability = get_gpu_compute_capability(net->gpu_index, device_name);
		#ifdef CUDNN_HALF
		if (compute_capability >= 700) net->cudnn_half = 1;
		else net->cudnn_half = 0;
		#endif// CUDNN_HALF
		fprintf(stderr, " %d : compute_capability = %d, cudnn_half = %d, GPU: %s \n", net->gpu_index, compute_capability, net->cudnn_half, device_name);
	}
	else fprintf(stderr, " GPU isn't used \n");
	#endif// GPU

	if (settings.policy == ELearningRatePolicy::kStep)
	{
		settings.step	= net.i("step"	, 1		);
		settings.scale	= net.f("scale"	, 1.0f	);
	}
	else if (settings.policy == ELearningRatePolicy::kSteps or settings.policy == ELearningRatePolicy::kSGDR)
	{
		steps		= net.vi("steps"		);	// "steps" is often 2 ints, such as:  steps=4000,6000
		scales		= net.vf("scales"		);	// "scales" is often 2 floats, such as:  scales=.1,.1
		seq_scales	= net.vf("seq_scales"	);	// "seq_scales" isn't used in any of the configs I see available
		settings.num_steps = steps.size();

		if (settings.policy == ELearningRatePolicy::kSteps and (steps.empty() or scales.empty()))
		{
			throw std::runtime_error("\"Steps\" policy must have \"steps\" and \"scales\" in .cfg file");
		}

		// make sure "scales" and "seq_scales" have exactly the same number of entries as "steps"
		scales		.resize(settings.num_steps, 1.0f);
		seq_scales	.resize(settings.num_steps, 1.0f);
	}
	else if (settings.policy == ELearningRatePolicy::kEXP)
	{
		settings.gamma = net.f("gamma", 1.0f);
	}
	else if (settings.policy == ELearningRatePolicy::kSigmoid)
	{
		settings.step	= net.i("step"	, 1		);
		settings.gamma	= net.f("gamma"	, 1.0f	);
	}
	else if (settings.policy == ELearningRatePolicy::kPoly or settings.policy == ELearningRatePolicy::kRandom)
	{
//		settings.power = net.f("power", 1.0f);
	}

	return *this;
}


#if 0
Darknet_ng::Network *Darknet_ng::load_network_custom(char *cfg, char *weights, int clear, int batch)
{
	printf(" Try to load cfg: %s, weights: %s, clear = %d \n", cfg, weights, clear);

	Network * net = (Network*)xcalloc(1, sizeof(network));

	*net = parse_network_cfg_custom(cfg, batch, 1);
	if (weights && weights[0] != 0) {
		printf(" Try to load weights: %s \n", weights);
		load_weights(net, weights);
	}
	fuse_conv_batchnorm(*net);
	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


Darknet_ng::Network *Darknet_ng::load_network(char *cfg, char *weights, int clear)
{
	printf(" Try to load cfg: %s, clear = %d \n", cfg, clear);
	network* net = (network*)xcalloc(1, sizeof(network));
	*net = parse_network_cfg(cfg);
	if (weights && weights[0] != 0) {
		printf(" Try to load weights: %s \n", weights);
		load_weights(net, weights);
	}
	if (clear) {
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}
	return net;
}
#endif


#if 0 /// @todo
Darknet_ng::Network Darknet_ng::load_network(const std::filesystem::path & cfg_filename, const std::filesystem::path & weights_filename, const bool clear)
{
	std::cout
		<< "Network config .... " << cfg_filename		.string() << std::endl
		<< "Network weights ... " << weights_filename	.string() << std::endl;

	Network net(cfg_filename);
//	Network net = parse_network_cfg_custom(cfg_filename, 1, 1);

#ifdef WORK_IN_PROGRESS /// @todo
//	Network net = parse_network_cfg_custom(cfg_filename, batch, 1);

	load_weights(net, weights_filename);

	fuse_conv_batchnorm(net);

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}
#endif
	return net;
}


Darknet_ng::Network Darknet_ng::parse_network_cfg_custom(const std::filesystem::path & cfg_filename, int batch, int time_steps)
{
	Config cfg(cfg_filename);
	if (cfg.empty())
	{
		throw std::invalid_argument("configuration file is empty: " + cfg_filename.string());
	}

#ifdef WORK_IN_PROGRESS_DONE /// @todo
	list *sections = read_cfg(cfg_filename);
	node *n = sections->front;

	if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
	net.gpu_index = gpu_index;

#ifdef WORK_IN_PROGRESS /// @todo

	Network net = make_network(sections->size - 1);
	size_params params;

	if (batch > 0) params.train = 0;    // allocates memory for Inference only
	else params.train = 1;              // allocates memory for Inference & Training

	section *s = (section *)n->val;
	list *options = s->options;
	parse_net_options(options, &net);

	#ifdef GPU
	printf("net.optimized_memory = %d \n", net.optimized_memory);
	if (net.optimized_memory >= 2 && params.train) {
		pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);   // pre-allocate 8 GB CPU-RAM for pinned memory
	}
	#endif  // GPU

	params.h = net.h;
	params.w = net.w;
	params.c = net.c;
	params.inputs = net.inputs;
	if (batch > 0) net.batch = batch;
	if (time_steps > 0) net.time_steps = time_steps;
	if (net.batch < 1) net.batch = 1;
	if (net.time_steps < 1) net.time_steps = 1;
	if (net.batch < net.time_steps) net.batch = net.time_steps;
	params.batch = net.batch;
	params.time_steps = net.time_steps;
	params.net = net;
	printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

	int last_stop_backward = -1;
	int avg_outputs = 0;
	int avg_counter = 0;
	float bflops = 0;
	size_t workspace_size = 0;
	size_t max_inputs = 0;
	size_t max_outputs = 0;
	int receptive_w = 1, receptive_h = 1;
	int receptive_w_scale = 1, receptive_h_scale = 1;
	const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);

	n = n->next;
	int count = 0;
	free_section(s);

	// find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
	node *n_tmp = n;
	int count_tmp = 0;
	if (params.train == 1) {
		while (n_tmp) {
			s = (section *)n_tmp->val;
			options = s->options;
			int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
			if (stopbackward == 1) {
				last_stop_backward = count_tmp;
				printf("last_stop_backward = %d \n", last_stop_backward);
			}
			n_tmp = n_tmp->next;
			++count_tmp;
		}
	}

	int old_params_train = params.train;

	fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
	while(n){

		params.train = old_params_train;
		if (count < last_stop_backward) params.train = 0;

		params.index = count;
		fprintf(stderr, "%4d ", count);
		s = (section *)n->val;
		options = s->options;
		layer l = { (LAYER_TYPE)0 };
		LAYER_TYPE lt = string_to_layer_type(s->type);
		if(lt == CONVOLUTIONAL){
			l = parse_convolutional(options, params);
		}else if(lt == LOCAL){
			l = parse_local(options, params);
		}else if(lt == ACTIVE){
			l = parse_activation(options, params);
		}else if(lt == RNN){
			l = parse_rnn(options, params);
		}else if(lt == GRU){
			l = parse_gru(options, params);
		}else if(lt == LSTM){
			l = parse_lstm(options, params);
		}else if (lt == CONV_LSTM) {
			l = parse_conv_lstm(options, params);
		}else if (lt == HISTORY) {
			l = parse_history(options, params);
		}else if(lt == CRNN){
			l = parse_crnn(options, params);
		}else if(lt == CONNECTED){
			l = parse_connected(options, params);
		}else if(lt == CROP){
			l = parse_crop(options, params);
		}else if(lt == COST){
			l = parse_cost(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == REGION){
			l = parse_region(options, params);
			l.keep_delta_gpu = 1;
		}else if (lt == YOLO) {
			l = parse_yolo(options, params);
			l.keep_delta_gpu = 1;
		}else if (lt == GAUSSIAN_YOLO) {
			l = parse_gaussian_yolo(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == DETECTION){
			l = parse_detection(options, params);
		}else if(lt == SOFTMAX){
			l = parse_softmax(options, params);
			net.hierarchy = l.softmax_tree;
			l.keep_delta_gpu = 1;
		}else if (lt == CONTRASTIVE) {
			l = parse_contrastive(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == NORMALIZATION){
			l = parse_normalization(options, params);
		}else if(lt == BATCHNORM){
			l = parse_batchnorm(options, params);
		}else if(lt == MAXPOOL){
			l = parse_maxpool(options, params);
		}else if (lt == LOCAL_AVGPOOL) {
			l = parse_local_avgpool(options, params);
		}else if(lt == REORG){
			l = parse_reorg(options, params);        }
			else if (lt == REORG_OLD) {
				l = parse_reorg_old(options, params);
			}else if(lt == AVGPOOL){
				l = parse_avgpool(options, params);
			}else if(lt == ROUTE){
				l = parse_route(options, params);
				int k;
				for (k = 0; k < l.n; ++k) {
					net.layers[l.input_layers[k]].use_bin_output = 0;
					if (count >= last_stop_backward)
						net.layers[l.input_layers[k]].keep_delta_gpu = 1;
				}
			}else if (lt == UPSAMPLE) {
				l = parse_upsample(options, params, net);
			}else if(lt == SHORTCUT){
				l = parse_shortcut(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				if (count >= last_stop_backward)
					net.layers[l.index].keep_delta_gpu = 1;
			}else if (lt == SCALE_CHANNELS) {
				l = parse_scale_channels(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
			}
			else if (lt == SAM) {
				l = parse_sam(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
			} else if (lt == IMPLICIT) {
				l = parse_implicit(options, params, net);
			}else if(lt == DROPOUT){
				l = parse_dropout(options, params);
				l.output = net.layers[count-1].output;
				l.delta = net.layers[count-1].delta;
				#ifdef GPU
				l.output_gpu = net.layers[count-1].output_gpu;
				l.delta_gpu = net.layers[count-1].delta_gpu;
				l.keep_delta_gpu = 1;
				#endif
			}
			else if (lt == EMPTY) {
				layer empty_layer = {(LAYER_TYPE)0};
				l = empty_layer;
				l.type = EMPTY;
				l.w = l.out_w = params.w;
				l.h = l.out_h = params.h;
				l.c = l.out_c = params.c;
				l.batch = params.batch;
				l.inputs = l.outputs = params.inputs;
				l.output = net.layers[count - 1].output;
				l.delta = net.layers[count - 1].delta;
				l.forward = empty_func;
				l.backward = empty_func;
				#ifdef GPU
				l.output_gpu = net.layers[count - 1].output_gpu;
				l.delta_gpu = net.layers[count - 1].delta_gpu;
				l.keep_delta_gpu = 1;
				l.forward_gpu = empty_func;
				l.backward_gpu = empty_func;
				#endif
				fprintf(stderr, "empty \n");
			}else{
				fprintf(stderr, "Type not recognized: %s\n", s->type);
			}

			// calculate receptive field
			if(show_receptive_field)
			{
				int dilation = max_val_cmp(1, l.dilation);
				int stride = max_val_cmp(1, l.stride);
				int size = max_val_cmp(1, l.size);

				if (l.type == UPSAMPLE || (l.type == REORG))
				{

					l.receptive_w = receptive_w;
					l.receptive_h = receptive_h;
					l.receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
					l.receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;

				}
				else {
					if (l.type == ROUTE) {
						receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
						int k;
						for (k = 0; k < l.n; ++k) {
							layer route_l = net.layers[l.input_layers[k]];
							receptive_w = max_val_cmp(receptive_w, route_l.receptive_w);
							receptive_h = max_val_cmp(receptive_h, route_l.receptive_h);
							receptive_w_scale = max_val_cmp(receptive_w_scale, route_l.receptive_w_scale);
							receptive_h_scale = max_val_cmp(receptive_h_scale, route_l.receptive_h_scale);
						}
					}
					else
					{
						int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
						increase_receptive = max_val_cmp(0, increase_receptive);

						receptive_w += increase_receptive * receptive_w_scale;
						receptive_h += increase_receptive * receptive_h_scale;
						receptive_w_scale *= stride;
						receptive_h_scale *= stride;
					}

					l.receptive_w = receptive_w;
					l.receptive_h = receptive_h;
					l.receptive_w_scale = receptive_w_scale;
					l.receptive_h_scale = receptive_h_scale;
				}
				//printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d, receptive_w_scale = %d - ", size, dilation, stride, receptive_w, receptive_w_scale);

				int cur_receptive_w = receptive_w;
				int cur_receptive_h = receptive_h;

				fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
			}

			#ifdef GPU
			// futher GPU-memory optimization: net.optimized_memory == 2
			l.optimized_memory = net.optimized_memory;
			if (net.optimized_memory == 1 && params.train && l.type != DROPOUT) {
				if (l.delta_gpu) {
					cuda_free(l.delta_gpu);
					l.delta_gpu = NULL;
				}
			} else if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
			{
				if (l.output_gpu) {
					cuda_free(l.output_gpu);
					//l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
					l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}
				if (l.activation_input_gpu) {
					cuda_free(l.activation_input_gpu);
					l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}

				if (l.x_gpu) {
					cuda_free(l.x_gpu);
					l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}

				// maximum optimization
				if (net.optimized_memory >= 3 && l.type != DROPOUT) {
					if (l.delta_gpu) {
						cuda_free(l.delta_gpu);
						//l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
						//printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
					}
				}

				if (l.type == CONVOLUTIONAL) {
					set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
				}
			}
			#endif // GPU

			l.clip = option_find_float_quiet(options, "clip", 0);
			l.dynamic_minibatch = net.dynamic_minibatch;
			l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
			l.dont_update = option_find_int_quiet(options, "dont_update", 0);
			l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
			l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
			l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
			l.dontload = option_find_int_quiet(options, "dontload", 0);
			l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
			l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
			option_unused(options);

			if (l.stopbackward == 1) printf(" ------- previous layers are frozen ------- \n");

			net.layers[count] = l;
			if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
			if (l.inputs > max_inputs) max_inputs = l.inputs;
			if (l.outputs > max_outputs) max_outputs = l.outputs;
			free_section(s);
			n = n->next;
			++count;
			if(n){
				if (l.antialiasing) {
					params.h = l.input_layer->out_h;
					params.w = l.input_layer->out_w;
					params.c = l.input_layer->out_c;
					params.inputs = l.input_layer->outputs;
				}
				else {
					params.h = l.out_h;
					params.w = l.out_w;
					params.c = l.out_c;
					params.inputs = l.outputs;
				}
			}
			if (l.bflops > 0) bflops += l.bflops;

			if (l.w > 1 && l.h > 1) {
				avg_outputs += l.outputs;
				avg_counter++;
			}
	}

	if (last_stop_backward > -1) {
		int k;
		for (k = 0; k < last_stop_backward; ++k) {
			layer l = net.layers[k];
			if (l.keep_delta_gpu) {
				if (!l.delta) {
					net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
				}
				#ifdef GPU
				if (!l.delta_gpu) {
					net.layers[k].delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
				}
				#endif
			}

			net.layers[k].onlyforward = 1;
			net.layers[k].train = 0;
		}
	}

	free_list(sections);

	#ifdef GPU
	if (net.optimized_memory && params.train)
	{
		int k;
		for (k = 0; k < net.n; ++k) {
			layer l = net.layers[k];
			// delta GPU-memory optimization: net.optimized_memory == 1
			if (!l.keep_delta_gpu) {
				const size_t delta_size = l.outputs*l.batch; // l.steps
				if (net.max_delta_gpu_size < delta_size) {
					net.max_delta_gpu_size = delta_size;
					if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
					if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
					assert(net.max_delta_gpu_size > 0);
					net.global_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
					net.state_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
				}
				if (l.delta_gpu) {
					if (net.optimized_memory >= 3) {}
					else cuda_free(l.delta_gpu);
				}
				l.delta_gpu = net.global_delta_gpu;
			}
			else {
				if (!l.delta_gpu) l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
			}

			// maximum optimization
			if (net.optimized_memory >= 3 && l.type != DROPOUT) {
				if (l.delta_gpu && l.keep_delta_gpu) {
					//cuda_free(l.delta_gpu);   // already called above
					l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
					//printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
				}
			}

			net.layers[k] = l;
		}
	}
	#endif

	set_train_only_bn(net); // set l.train_only_bn for all required layers

	net.outputs = get_network_output_size(net);
	net.output = get_network_output(net);
	avg_outputs = avg_outputs / avg_counter;
	fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
	fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
	#ifdef GPU
	get_cuda_stream();
	//get_cuda_memcpy_stream();
	if (gpu_index >= 0)
	{
		int size = get_network_input_size(net) * net.batch;
		net.input_state_gpu = cuda_make_array(0, size);
		if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
		else {
			cudaGetLastError(); // reset CUDA-error
			net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
		}

		// pre-allocate memory for inference on Tensor Cores (fp16)
		*net.max_input16_size = 0;
		*net.max_output16_size = 0;
		if (net.cudnn_half) {
			*net.max_input16_size = max_inputs;
			CHECK_CUDA(cudaMalloc((void **)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
			*net.max_output16_size = max_outputs;
			CHECK_CUDA(cudaMalloc((void **)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
		}
		if (workspace_size) {
			fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n", (float)workspace_size/1000000);
			net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
		}
		else {
			net.workspace = (float*)xcalloc(1, workspace_size);
		}
	}
	#else
	if (workspace_size) {
		net.workspace = (float*)xcalloc(1, workspace_size);
	}
	#endif

	LAYER_TYPE lt = net.layers[net.n - 1].type;
	if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
		printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
			   net.w, net.h);
	}
#endif
#endif
	Network net;
	return net;
}


Darknet_ng::Network make_network(int n)
{
	Network net = {0};
	net.n = n;
	net.layers = (layer*)xcalloc(net.n, sizeof(layer));
	net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
	net.cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
	net.badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_max = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_std = (float*)xcalloc(1, sizeof(float));
	net.cur_iteration = (int*)xcalloc(1, sizeof(int));
	net.total_bbox = (int*)xcalloc(1, sizeof(int));
	net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
	*net.rewritten_bbox = *net.total_bbox = 0;
	#ifdef GPU
	net.input_gpu = (float**)xcalloc(1, sizeof(float*));
	net.truth_gpu = (float**)xcalloc(1, sizeof(float*));

	net.input16_gpu = (float**)xcalloc(1, sizeof(float*));
	net.output16_gpu = (float**)xcalloc(1, sizeof(float*));
	net.max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
	net.max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
	#endif
	return net;
}
#endif
