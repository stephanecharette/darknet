// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include <cmath>


Darknet_ng::Network & Darknet_ng::Network::parse_convolutional(const Darknet_ng::Section & section, const size_t layer_index)
{
	// was:  convolutional_layer parse_convolutional(list *options, size_params params);

	const int n						= section.i("filters"				, 1);
	const int groups				= section.i("groups"				, 1);
	const int size					= section.i("size"					, 1);
	const int stride				= section.i("stride"				, 1);
	int stride_x					= section.i("stride_x"				, -1);
	int stride_y					= section.i("stride_y"				, -1);
	if (stride_x < 1)				stride_x = stride;
	if (stride_y < 1)				stride_y = stride;
	int dilation					= section.i("dilation"				, 1);
	if (size == 1)					dilation = 1;
	const int antialiasing			= section.i("antialiasing"			, 0);
	const int pad					= section.i("pad"					, 0);
	int padding						= section.i("padding"				, 0);
	if (pad)						padding = size / 2;
	const auto activation			= activation_from_string(section.s("activation", "logistic"));

	/// @todo why does this get a float but store it in an int?  Should this be a float?
	const int assisted_excitation	= section.f("assisted_excitation"	, 0.0f);

	int batch_normalize				= section.i("batch_normalize"		, 0);
	const int cbn					= section.i("cbn"					, 0);
	if (cbn)						batch_normalize = 2;
	const int binary				= section.i("binary"				, 0);
	const int xnor					= section.i("xnor"					, 0);
	const int use_bin_output		= section.i("bin_output"			, 0);
	const int sway					= section.i("sway"					, 0); ///< @todo should this be a bool?
	const int rotate				= section.i("rotate"				, 0); ///< @todo should this be a bool?
	const int stretch				= section.i("stretch"				, 0); ///< @todo should this be a bool?
	const int stretch_sway			= section.i("stretch_sway"			, 0); ///< @todo should this be a bool?

	const int deform = sway + rotate + stretch + stretch_sway;
	if (deform < 0 or deform > 1)
	{
		throw std::invalid_argument("convolutional layer at line #" + std::to_string(section.line_number) + " must only enable a maximum of 1 of sway, rotate, stretch, or stretch_sway");
	}
	if (deform == 1 and size == 1)
	{
		throw std::invalid_argument("convolutional layer at line #" + std::to_string(section.line_number) + " must have a larger size to use sway, rotate, stretch, or stretch_sway");
	}

	const int share_index = section.i("share_index", -1000000000);
	Layer * share_layer = nullptr;
	if (share_index >= 0)
	{
		share_layer = &layers.at(share_index);
	}
	else if (share_index != -1000000000)
	{
		share_layer = &layers.at(layer_index + share_index);
	}

	Layer & layer = layers.at(layer_index);

	make_convolutional_layer(
		layer,
		settings.batch,
		1,
		settings.h,
		settings.w,
		settings.c,
		n,
		groups,
		size,
		stride_x,
		stride_y,
		dilation,
		padding,
		activation,
		batch_normalize,
		binary,
		xnor,
		settings.adam,
		use_bin_output,
		layer_index,
		antialiasing,
		share_layer,
		assisted_excitation,
		deform,
		settings.train);

	layer.sway				= sway;
	layer.rotate			= rotate;
	layer.stretch			= stretch;
	layer.stretch_sway		= stretch_sway;
	layer.flipped			= section.i("flipped"		, 0		);
	layer.dot				= section.f("dot"			, 0.0f	);
	layer.angle				= section.f("angle"			, 15.0f	);
	layer.grad_centr		= section.i("grad_centr"	, 0		);
	layer.reverse			= section.f("reverse"		, 0.0f	);
	layer.coordconv			= section.i("coordconv"		, 0		);
	layer.stream			= section.i("stream"		, -1	);
	layer.wait_stream_id	= section.i("wait_stream"	, -1	);

	if (settings.adam)
	{
		layer.B1	= settings.B1;
		layer.B2	= settings.B2;
		layer.eps	= settings.eps;
	}

	//	return layer;

	return *this;
}


// was: convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train);
Darknet_ng::Layer Darknet_ng::make_convolutional_layer(Darknet_ng::Layer & layer, int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, Darknet_ng::EActivation activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, Layer *share_layer, int assisted_excitation, int deform, int train)
{
	int total_batch = batch * steps;
	int i;
//	Layer layer = {0};// = { (LAYER_TYPE)0 };
//	layer = {0};
	layer.type = ELayerType::kConvolutional;
	layer.train = train;

	if (xnor) groups = 1;   // disable groups for XNOR-net
	if (groups < 1) groups = 1;

	const int blur_stride_x = stride_x;
	const int blur_stride_y = stride_y;
	layer.antialiasing = antialiasing;
	if (antialiasing)
	{
		stride_x = stride_y = layer.stride = layer.stride_x = layer.stride_y = 1; // use stride=1 in host-layer
	}

	layer.wait_stream_id = -1;
	layer.deform = deform;
	layer.assisted_excitation = assisted_excitation;
	layer.share_layer = share_layer;
	layer.index = index;
	layer.h = h;
	layer.w = w;
	layer.c = c;
	layer.groups = groups;
	layer.n = n;
	layer.binary = binary;
	layer.xnor = xnor;
	layer.use_bin_output = use_bin_output;
	layer.batch = batch;
	layer.steps = steps;
	layer.stride = stride_x;
	layer.stride_x = stride_x;
	layer.stride_y = stride_y;
	layer.dilation = dilation;
	layer.size = size;
	layer.pad = padding;
	layer.batch_normalize = batch_normalize;
	layer.learning_rate_scale = 1;
	layer.nweights = (c / groups) * n * size * size;

	if (layer.share_layer)
	{
		if (layer.size != layer.share_layer->size or layer.nweights != layer.share_layer->nweights or layer.c != layer.share_layer->c or layer.n != layer.share_layer->n)
		{
			printf(" Layer size, nweights, channels or filters don't match for the share_layer");
			getchar(); ///< @todo
		}

		layer.weights = layer.share_layer->weights;
		layer.weight_updates = layer.share_layer->weight_updates;

		layer.biases = layer.share_layer->biases;
		layer.bias_updates = layer.share_layer->bias_updates;
	}
	else
	{
		layer.weights = (float*)xcalloc(layer.nweights, sizeof(float));
		layer.biases = (float*)xcalloc(n, sizeof(float));

		if (train)
		{
			layer.weight_updates = (float*)xcalloc(layer.nweights, sizeof(float));
			layer.bias_updates = (float*)xcalloc(n, sizeof(float));

			layer.weights_ema = (float*)xcalloc(layer.nweights, sizeof(float));
			layer.biases_ema = (float*)xcalloc(n, sizeof(float));
		}
	}

	// float scale = 1./sqrt(size*size*c);
	float scale = std::sqrt(2.0f / (size * size * c / groups));
	if (layer.activation == EActivation::kNormCHAN			or
		layer.activation == EActivation::kNormCHANSoftmax	or
		layer.activation == EActivation::kNormCHANSoftmaxMaxVal)
	{
		for (i = 0; i < layer.nweights; ++i)
		{
			layer.weights[i] = 1;   // rand_normal();
		}
	}
	else
	{
		for (i = 0; i < layer.nweights; ++i)
		{
			layer.weights[i] = scale * rand_uniform(-1, 1);   // rand_normal();
		}
	}

	const int out_h		= convolutional_out_height(layer);
	const int out_w		= convolutional_out_width(layer);

	layer.out_h			= out_h;
	layer.out_w			= out_w;
	layer.out_c			= n;
	layer.outputs		= layer.out_h * layer.out_w * layer.out_c;
	layer.inputs		= layer.w * layer.h * layer.c;
	layer.activation	= activation;

	layer.output = (float*)xcalloc(total_batch*layer.outputs, sizeof(float));
	#ifndef GPU
	if (train)
	{
		layer.delta = (float*)xcalloc(total_batch * layer.outputs, sizeof(float));
	}
	#endif  // not GPU

#if STEPHANE /// @todo
	layer.forward	= forward_convolutional_layer;
	layer.backward	= backward_convolutional_layer;
	layer.update	= update_convolutional_layer;
#endif

	if (binary)
	{
		layer.binary_weights = (float*)xcalloc(layer.nweights, sizeof(float));
		layer.cweights = (char*)xcalloc(layer.nweights, sizeof(char));
		layer.scales = (float*)xcalloc(n, sizeof(float));
	}

	if (xnor)
	{
		layer.binary_weights = (float*)xcalloc(layer.nweights, sizeof(float));
		layer.binary_input = (float*)xcalloc(layer.inputs * layer.batch, sizeof(float));

		int align = 32;// 8;
		int src_align = layer.out_h * layer.out_w;
		layer.bit_align = src_align + (align - src_align % align);

		layer.mean_arr = (float*)xcalloc(layer.n, sizeof(float));

		const size_t new_c = layer.c / 32;
		size_t in_re_packed_input_size = new_c * layer.w * layer.h + 1;
		layer.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

		layer.lda_align = 256;  // AVX2
		int k = layer.size * layer.size * layer.c;
		size_t k_aligned = k + (layer.lda_align - k % layer.lda_align);
		size_t t_bit_input_size = k_aligned * layer.bit_align / 8;
		layer.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
	}

	if (batch_normalize)
	{
		if (layer.share_layer)
		{
			layer.scales			= layer.share_layer->scales;
			layer.scale_updates		= layer.share_layer->scale_updates;
			layer.mean				= layer.share_layer->mean;
			layer.variance			= layer.share_layer->variance;
			layer.mean_delta		= layer.share_layer->mean_delta;
			layer.variance_delta	= layer.share_layer->variance_delta;
			layer.rolling_mean		= layer.share_layer->rolling_mean;
			layer.rolling_variance	= layer.share_layer->rolling_variance;
		}
		else
		{
			layer.scales = (float*)xcalloc(n, sizeof(float));
			for (i = 0; i < n; ++i)
			{
				layer.scales[i] = 1;
			}
			if (train)
			{
				layer.scales_ema		= (float*)xcalloc(n, sizeof(float));
				layer.scale_updates		= (float*)xcalloc(n, sizeof(float));

				layer.mean				= (float*)xcalloc(n, sizeof(float));
				layer.variance			= (float*)xcalloc(n, sizeof(float));

				layer.mean_delta		= (float*)xcalloc(n, sizeof(float));
				layer.variance_delta	= (float*)xcalloc(n, sizeof(float));
			}

			layer.rolling_mean		= (float*)xcalloc(n, sizeof(float));
			layer.rolling_variance	= (float*)xcalloc(n, sizeof(float));
		}

		#ifndef GPU
		if (train)
		{
			layer.x = (float*)xcalloc(total_batch * layer.outputs, sizeof(float));
			layer.x_norm = (float*)xcalloc(total_batch * layer.outputs, sizeof(float));
		}
		#endif  // not GPU
	}

	#ifndef GPU
	if (layer.activation == EActivation::kSWISH	or
		layer.activation == EActivation::kMISH	or
		layer.activation == EActivation::kHardMISH)
	{
		layer.activation_input = (float*)calloc(total_batch * layer.outputs, sizeof(float));
	}
	#endif  // not GPU

	if (adam)
	{
		layer.adam = 1;
		layer.m = (float*)xcalloc(layer.nweights, sizeof(float));
		layer.v = (float*)xcalloc(layer.nweights, sizeof(float));
		layer.bias_m = (float*)xcalloc(n, sizeof(float));
		layer.scale_m = (float*)xcalloc(n, sizeof(float));
		layer.bias_v = (float*)xcalloc(n, sizeof(float));
		layer.scale_v = (float*)xcalloc(n, sizeof(float));
	}

	#ifdef GPU

	layer.forward_gpu = forward_convolutional_layer_gpu;
	layer.backward_gpu = backward_convolutional_layer_gpu;
	layer.update_gpu = update_convolutional_layer_gpu;

	if(gpu_index >= 0)
	{
		if (train && (layer.activation == SWISH or layer.activation == MISH or layer.activation == HARD_MISH))
		{
			layer.activation_input_gpu = cuda_make_array(layer.activation_input, total_batch * layer.outputs);
		}

		if (layer.deform)
		{
			layer.weight_deform_gpu = cuda_make_array(NULL, layer.nweights);
		}

		if (adam)
		{
			layer.m_gpu = cuda_make_array(layer.m, layer.nweights);
			layer.v_gpu = cuda_make_array(layer.v, layer.nweights);
			layer.bias_m_gpu = cuda_make_array(layer.bias_m, n);
			layer.bias_v_gpu = cuda_make_array(layer.bias_v, n);
			layer.scale_m_gpu = cuda_make_array(layer.scale_m, n);
			layer.scale_v_gpu = cuda_make_array(layer.scale_v, n);
		}
		if (l.share_layer)
		{
			layer.weights_gpu = layer.share_layer->weights_gpu;
			layer.weight_updates_gpu = layer.share_layer->weight_updates_gpu;
			layer.weights_gpu16 = layer.share_layer->weights_gpu16;
			layer.weight_updates_gpu16 = layer.share_layer->weight_updates_gpu16;
			layer.biases_gpu = layer.share_layer->biases_gpu;
			layer.bias_updates_gpu = layer.share_layer->bias_updates_gpu;
		}
		else
		{
			layer.weights_gpu = cuda_make_array(layer.weights, layer.nweights);
			if (train)
			{
				layer.weight_updates_gpu = cuda_make_array(layer.weight_updates, layer.nweights);
			}
			#ifdef CUDNN_HALF
			layer.weights_gpu16 = cuda_make_array(NULL, layer.nweights / 2 + 1);
			if (train)
			{
				layer.weight_updates_gpu16 = cuda_make_array(NULL, layer.nweights / 2 + 1);
			}
			#endif  // CUDNN_HALF
			layer.biases_gpu = cuda_make_array(layer.biases, n);
			if (train)
			{
				layer.bias_updates_gpu = cuda_make_array(layer.bias_updates, n);
			}
		}

		layer.output_gpu = cuda_make_array(layer.output, total_batch * out_h * out_w * n);
		if (train)
		{
			layer.delta_gpu = cuda_make_array(layer.delta, total_batch * out_h * out_w * n);
		}

		if (binary)
		{
			layer.binary_weights_gpu = cuda_make_array(layer.weights, layer.nweights);
		}
		if (xnor)
		{
			layer.binary_weights_gpu = cuda_make_array(layer.weights, layer.nweights);
			layer.mean_arr_gpu = cuda_make_array(0, layer.n);
			layer.binary_input_gpu = cuda_make_array(0, layer.inputs * layer.batch);
		}

		if (batch_normalize)
		{
			if (layer.share_layer)
			{
				layer.scales_gpu = layer.share_layer->scales_gpu;
				layer.scale_updates_gpu = layer.share_layer->scale_updates_gpu;
				layer.mean_gpu = layer.share_layer->mean_gpu;
				layer.variance_gpu = layer.share_layer->variance_gpu;
				layer.rolling_mean_gpu = layer.share_layer->rolling_mean_gpu;
				layer.rolling_variance_gpu = layer.share_layer->rolling_variance_gpu;
				layer.mean_delta_gpu = layer.share_layer->mean_delta_gpu;
				layer.variance_delta_gpu = layer.share_layer->variance_delta_gpu;
			}
			else
			{
				layer.scales_gpu = cuda_make_array(layer.scales, n);

				if (train)
				{
					layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, n);

					layer.mean_gpu = cuda_make_array(layer.mean, n);
					layer.variance_gpu = cuda_make_array(layer.variance, n);
					layer.m_cbn_avg_gpu = cuda_make_array(layer.mean, n);
					layer.v_cbn_avg_gpu = cuda_make_array(layer.variance, n);
					#ifndef CUDNN
					layer.mean_delta_gpu = cuda_make_array(layer.mean, n);
					layer.variance_delta_gpu = cuda_make_array(layer.variance, n);
					#endif  // CUDNN
				}

				layer.rolling_mean_gpu = cuda_make_array(layer.mean, n);
				layer.rolling_variance_gpu = cuda_make_array(layer.variance, n);
			}

			if (train)
			{
				layer.x_gpu = cuda_make_array(l.output, total_batch * out_h * out_w  *n);
				#ifndef CUDNN
				layer.x_norm_gpu = cuda_make_array(layer.output, total_batch * out_h * out_w * n);
				#endif  // CUDNN
			}
		}

		if (layer.assisted_excitation)
		{
			const int size = layer.out_w * layer.out_h * layer.batch;
			layer.gt_gpu = cuda_make_array(NULL, size);
			layer.a_avg_gpu = cuda_make_array(NULL, size);
		}
		#ifdef CUDNN
		create_convolutional_cudnn_tensors(&layer);
		cudnn_convolutional_setup(&layer, cudnn_fastest, 0);
		#endif  // CUDNN
	}
	#endif  // GPU
	layer.workspace_size = get_convolutional_workspace_size(layer);

	//fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
	layer.bflops = (2.0 * layer.nweights * layer.out_h * layer.out_w) / 1000000000.0f;
	if (layer.xnor)
	{
		layer.bflops = layer.bflops / 32;
	}
	if (layer.xnor and layer.use_bin_output)
	{
		fprintf(stderr, "convXB");
	}
	else if (layer.xnor)
	{
		fprintf(stderr, "convX ");
	}
	else if (layer.share_layer)
	{
		fprintf(stderr, "convS ");
	}
	else if (layer.assisted_excitation)
	{
		fprintf(stderr, "convAE");
	}
	else
	{
		fprintf(stderr, "conv  ");
	}

	if (groups > 1)
	{
		fprintf(stderr, "%5d/%4d ", n, groups);
	}
	else
	{
		fprintf(stderr, "%5d      ", n);
	}

	if (stride_x != stride_y)
	{
		fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
	}
	else
	{
		if (dilation > 1)
		{
			fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
		}
		else
		{
			fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
		}
	}

	fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, layer.out_w, layer.out_h, layer.out_c, layer.bflops);

	//fprintf(stderr, "%5d/%2d %2d x%2d /%2d(%d)%4d x%4d x%4d  -> %4d x%4d x%4d %5.3f BF\n", n, groups, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

	if (layer.antialiasing)
	{
		printf("AA:  ");
		layer.input_layer = (Layer*)calloc(1, sizeof(Layer)); // I think this line had a bug in the memory it would allocate
		int blur_size = 3;
		int blur_pad = blur_size / 2;
		if (layer.antialiasing == 2)
		{
			blur_size = 2;
			blur_pad = 0;
		}
		make_convolutional_layer(*layer.input_layer, batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, EActivation::kLinear, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
		const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
		int i;
		if (blur_size == 2)
		{
			for (i = 0; i < blur_nweights; i += (blur_size*blur_size))
			{
				layer.input_layer->weights[i + 0] = 1 / 4.f;
				layer.input_layer->weights[i + 1] = 1 / 4.f;
				layer.input_layer->weights[i + 2] = 1 / 4.f;
				layer.input_layer->weights[i + 3] = 1 / 4.f;
			}
		}
		else
		{
			for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
				layer.input_layer->weights[i + 0] = 1 / 16.0f;
				layer.input_layer->weights[i + 1] = 2 / 16.0f;
				layer.input_layer->weights[i + 2] = 1 / 16.0f;

				layer.input_layer->weights[i + 3] = 2 / 16.0f;
				layer.input_layer->weights[i + 4] = 4 / 16.0f;
				layer.input_layer->weights[i + 5] = 2 / 16.0f;

				layer.input_layer->weights[i + 6] = 1 / 16.0f;
				layer.input_layer->weights[i + 7] = 2 / 16.0f;
				layer.input_layer->weights[i + 8] = 1 / 16.0f;
			}
		}

		for (i = 0; i < n; ++i)
		{
			layer.input_layer->biases[i] = 0;
		}

		#ifdef GPU
		if (gpu_index >= 0)
		{
			layer.input_antialiasing_gpu = cuda_make_array(NULL, layer.batch * layer.outputs);
			push_convolutional_layer(*(layer.input_layer));
		}
		#endif  // GPU
	}

	return layer;
}


int Darknet_ng::convolutional_out_width(const Layer & layer)
{
	return (layer.w + 2 * layer.pad - layer.size) / layer.stride_x + 1;
}


int Darknet_ng::convolutional_out_height(const Layer & layer)
{
	return (layer.h + 2 * layer.pad - layer.size) / layer.stride_y + 1;
}


size_t Darknet_ng::get_convolutional_workspace_size(const Layer & layer)
{
	size_t workspace_size	= get_workspace_size32(layer);
	size_t workspace_size16	= get_workspace_size16(layer);
	if (workspace_size16 > workspace_size)
	{
		workspace_size = workspace_size16;
	}

	return workspace_size;
}
