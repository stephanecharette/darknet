// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include "config.hpp"
#include <iostream>


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


Darknet_ng::Network Darknet_ng::load_network(const std::filesystem::path & cfg_filename, const std::filesystem::path & weights_filename, const bool clear)
{
	std::cout
		<< "Network config .... " << cfg_filename		.string() << std::endl
		<< "Network weights ... " << weights_filename	.string() << std::endl;

	Network net = parse_network_cfg_custom(cfg_filename, 1, 1);

#ifdef STEPHANE
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
//	const auto foo = read_cfg(cfg_filename);

#ifdef STEPHANE
	list *sections = read_cfg(cfg_filename);
	node *n = sections->front;
	if(!n) error("Config file has no sections", DARKNET_LOC);
	Network net = make_network(sections->size - 1);
	net.gpu_index = gpu_index;
	size_params params;

	if (batch > 0) params.train = 0;    // allocates memory for Inference only
	else params.train = 1;              // allocates memory for Inference & Training

	section *s = (section *)n->val;
	list *options = s->options;
	if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
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
	Network net;
	return net;
}
