// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


Darknet_ng::Network & Darknet_ng::Network::parse_convolutional(const Darknet_ng::Section & section)
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
		throw std::runtime_error("convolutional layer at line #" + std::to_string(section.line_number) + " must only enable a maximum of 1 of sway, rotate, stretch, or stretch_sway");
	}
	if (deform == 1 and size == 1)
	{
		throw std::runtime_error("convolutional layer at line #" + std::to_string(section.line_number) + " must have a larger size to use sway, rotate, stretch, or stretch_sway");
	}

	#if 0 /// @todo
	const int share_index = section.i("share_index", -1000000000);
	convolutional_layer *share_layer = NULL;
	if(share_index >= 0) share_layer = &params.net.layers[share_index];
	else if(share_index != -1000000000) share_layer = &params.net.layers[params.index + share_index];
	#endif

#if 0
	//	convolutional_layer layer =
	Layer layer =
	make_convolutional_layer(
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
		params.index,
		antialiasing,
		share_layer,
		assisted_excitation,
		deform,
		settings.train);
#else
	Layer layer;
#endif

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


// convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train);
Darknet_ng::Layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train)
{
	int total_batch = batch*steps;
	int i;
	convolutional_layer l = { (LAYER_TYPE)0 };
	l.type = CONVOLUTIONAL;
	l.train = train;

	if (xnor) groups = 1;   // disable groups for XNOR-net
	if (groups < 1) groups = 1;

	const int blur_stride_x = stride_x;
	const int blur_stride_y = stride_y;
	l.antialiasing = antialiasing;
	if (antialiasing) {
		stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
	}

	l.wait_stream_id = -1;
	l.deform = deform;
	l.assisted_excitation = assisted_excitation;
	l.share_layer = share_layer;
	l.index = index;
	l.h = h;
	l.w = w;
	l.c = c;
	l.groups = groups;
	l.n = n;
	l.binary = binary;
	l.xnor = xnor;
	l.use_bin_output = use_bin_output;
	l.batch = batch;
	l.steps = steps;
	l.stride = stride_x;
	l.stride_x = stride_x;
	l.stride_y = stride_y;
	l.dilation = dilation;
	l.size = size;
	l.pad = padding;
	l.batch_normalize = batch_normalize;
	l.learning_rate_scale = 1;
	l.nweights = (c / groups) * n * size * size;

	if (l.share_layer) {
		if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n) {
			printf(" Layer size, nweights, channels or filters don't match for the share_layer");
			getchar();
		}

		l.weights = l.share_layer->weights;
		l.weight_updates = l.share_layer->weight_updates;

		l.biases = l.share_layer->biases;
		l.bias_updates = l.share_layer->bias_updates;
	}
	else {
		l.weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.biases = (float*)xcalloc(n, sizeof(float));

		if (train) {
			l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
			l.bias_updates = (float*)xcalloc(n, sizeof(float));

			l.weights_ema = (float*)xcalloc(l.nweights, sizeof(float));
			l.biases_ema = (float*)xcalloc(n, sizeof(float));
		}
	}

	// float scale = 1./sqrt(size*size*c);
	float scale = sqrt(2./(size*size*c/groups));
	if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) {
		for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;   // rand_normal();
	}
	else {
		for (i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
	}
	int out_h = convolutional_out_height(l);
	int out_w = convolutional_out_width(l);
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.activation = activation;

	l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
	#ifndef GPU
	if (train) l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
	#endif  // not GPU

	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;
	if(binary){
		l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.cweights = (char*)xcalloc(l.nweights, sizeof(char));
		l.scales = (float*)xcalloc(n, sizeof(float));
	}
	if(xnor){
		l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.binary_input = (float*)xcalloc(l.inputs * l.batch, sizeof(float));

		int align = 32;// 8;
		int src_align = l.out_h*l.out_w;
		l.bit_align = src_align + (align - src_align % align);

		l.mean_arr = (float*)xcalloc(l.n, sizeof(float));

		const size_t new_c = l.c / 32;
		size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
		l.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

		l.lda_align = 256;  // AVX2
		int k = l.size*l.size*l.c;
		size_t k_aligned = k + (l.lda_align - k%l.lda_align);
		size_t t_bit_input_size = k_aligned * l.bit_align / 8;
		l.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
	}

	if(batch_normalize){
		if (l.share_layer) {
			l.scales = l.share_layer->scales;
			l.scale_updates = l.share_layer->scale_updates;
			l.mean = l.share_layer->mean;
			l.variance = l.share_layer->variance;
			l.mean_delta = l.share_layer->mean_delta;
			l.variance_delta = l.share_layer->variance_delta;
			l.rolling_mean = l.share_layer->rolling_mean;
			l.rolling_variance = l.share_layer->rolling_variance;
		}
		else {
			l.scales = (float*)xcalloc(n, sizeof(float));
			for (i = 0; i < n; ++i) {
				l.scales[i] = 1;
			}
			if (train) {
				l.scales_ema = (float*)xcalloc(n, sizeof(float));
				l.scale_updates = (float*)xcalloc(n, sizeof(float));

				l.mean = (float*)xcalloc(n, sizeof(float));
				l.variance = (float*)xcalloc(n, sizeof(float));

				l.mean_delta = (float*)xcalloc(n, sizeof(float));
				l.variance_delta = (float*)xcalloc(n, sizeof(float));
			}
			l.rolling_mean = (float*)xcalloc(n, sizeof(float));
			l.rolling_variance = (float*)xcalloc(n, sizeof(float));
		}

		#ifndef GPU
		if (train) {
			l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
			l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
		}
		#endif  // not GPU
	}

	#ifndef GPU
	if (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH) l.activation_input = (float*)calloc(total_batch*l.outputs, sizeof(float));
	#endif  // not GPU

	if(adam){
		l.adam = 1;
		l.m = (float*)xcalloc(l.nweights, sizeof(float));
		l.v = (float*)xcalloc(l.nweights, sizeof(float));
		l.bias_m = (float*)xcalloc(n, sizeof(float));
		l.scale_m = (float*)xcalloc(n, sizeof(float));
		l.bias_v = (float*)xcalloc(n, sizeof(float));
		l.scale_v = (float*)xcalloc(n, sizeof(float));
	}

	#ifdef GPU


	l.forward_gpu = forward_convolutional_layer_gpu;
	l.backward_gpu = backward_convolutional_layer_gpu;
	l.update_gpu = update_convolutional_layer_gpu;

	if(gpu_index >= 0){

		if (train && (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH)) {
			l.activation_input_gpu = cuda_make_array(l.activation_input, total_batch*l.outputs);
		}

		if (l.deform) l.weight_deform_gpu = cuda_make_array(NULL, l.nweights);

		if (adam) {
			l.m_gpu = cuda_make_array(l.m, l.nweights);
			l.v_gpu = cuda_make_array(l.v, l.nweights);
			l.bias_m_gpu = cuda_make_array(l.bias_m, n);
			l.bias_v_gpu = cuda_make_array(l.bias_v, n);
			l.scale_m_gpu = cuda_make_array(l.scale_m, n);
			l.scale_v_gpu = cuda_make_array(l.scale_v, n);
		}
		if (l.share_layer) {
			l.weights_gpu = l.share_layer->weights_gpu;
			l.weight_updates_gpu = l.share_layer->weight_updates_gpu;
			l.weights_gpu16 = l.share_layer->weights_gpu16;
			l.weight_updates_gpu16 = l.share_layer->weight_updates_gpu16;
			l.biases_gpu = l.share_layer->biases_gpu;
			l.bias_updates_gpu = l.share_layer->bias_updates_gpu;
		}
		else {
			l.weights_gpu = cuda_make_array(l.weights, l.nweights);
			if (train) l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
			#ifdef CUDNN_HALF
			l.weights_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
			if (train) l.weight_updates_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
			#endif  // CUDNN_HALF
			l.biases_gpu = cuda_make_array(l.biases, n);
			if (train) l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
		}

		l.output_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
		if (train) l.delta_gpu = cuda_make_array(l.delta, total_batch*out_h*out_w*n);

		if(binary){
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
		}
		if(xnor){
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
			l.mean_arr_gpu = cuda_make_array(0, l.n);
			l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
		}

		if(batch_normalize){
			if (l.share_layer) {
				l.scales_gpu = l.share_layer->scales_gpu;
				l.scale_updates_gpu = l.share_layer->scale_updates_gpu;
				l.mean_gpu = l.share_layer->mean_gpu;
				l.variance_gpu = l.share_layer->variance_gpu;
				l.rolling_mean_gpu = l.share_layer->rolling_mean_gpu;
				l.rolling_variance_gpu = l.share_layer->rolling_variance_gpu;
				l.mean_delta_gpu = l.share_layer->mean_delta_gpu;
				l.variance_delta_gpu = l.share_layer->variance_delta_gpu;
			}
			else {
				l.scales_gpu = cuda_make_array(l.scales, n);

				if (train) {
					l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

					l.mean_gpu = cuda_make_array(l.mean, n);
					l.variance_gpu = cuda_make_array(l.variance, n);
					l.m_cbn_avg_gpu = cuda_make_array(l.mean, n);
					l.v_cbn_avg_gpu = cuda_make_array(l.variance, n);
					#ifndef CUDNN
					l.mean_delta_gpu = cuda_make_array(l.mean, n);
					l.variance_delta_gpu = cuda_make_array(l.variance, n);
					#endif  // CUDNN
				}

				l.rolling_mean_gpu = cuda_make_array(l.mean, n);
				l.rolling_variance_gpu = cuda_make_array(l.variance, n);
			}

			if (train) {
				l.x_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
				#ifndef CUDNN
				l.x_norm_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
				#endif  // CUDNN
			}
		}

		if (l.assisted_excitation)
		{
			const int size = l.out_w * l.out_h * l.batch;
			l.gt_gpu = cuda_make_array(NULL, size);
			l.a_avg_gpu = cuda_make_array(NULL, size);
		}
		#ifdef CUDNN
		create_convolutional_cudnn_tensors(&l);
		cudnn_convolutional_setup(&l, cudnn_fastest, 0);
		#endif  // CUDNN
	}
	#endif  // GPU
	l.workspace_size = get_convolutional_workspace_size(l);

	//fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
	l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
	if (l.xnor) l.bflops = l.bflops / 32;
	if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
	else if (l.xnor) fprintf(stderr, "convX ");
	else if (l.share_layer) fprintf(stderr, "convS ");
	else if (l.assisted_excitation) fprintf(stderr, "convAE");
	else fprintf(stderr, "conv  ");

	if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
	else           fprintf(stderr, "%5d      ", n);

	if (stride_x != stride_y) fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
	else {
		if (dilation > 1) fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
		else             fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
	}

	fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

	//fprintf(stderr, "%5d/%2d %2d x%2d /%2d(%d)%4d x%4d x%4d  -> %4d x%4d x%4d %5.3f BF\n", n, groups, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

	if (l.antialiasing) {
		printf("AA:  ");
		l.input_layer = (layer*)calloc(1, sizeof(layer));
		int blur_size = 3;
		int blur_pad = blur_size / 2;
		if (l.antialiasing == 2) {
			blur_size = 2;
			blur_pad = 0;
		}
		*(l.input_layer) = make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
		const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
		int i;
		if (blur_size == 2) {
			for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
				l.input_layer->weights[i + 0] = 1 / 4.f;
				l.input_layer->weights[i + 1] = 1 / 4.f;
				l.input_layer->weights[i + 2] = 1 / 4.f;
				l.input_layer->weights[i + 3] = 1 / 4.f;
			}
		}
		else {
			for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
				l.input_layer->weights[i + 0] = 1 / 16.f;
				l.input_layer->weights[i + 1] = 2 / 16.f;
				l.input_layer->weights[i + 2] = 1 / 16.f;

				l.input_layer->weights[i + 3] = 2 / 16.f;
				l.input_layer->weights[i + 4] = 4 / 16.f;
				l.input_layer->weights[i + 5] = 2 / 16.f;

				l.input_layer->weights[i + 6] = 1 / 16.f;
				l.input_layer->weights[i + 7] = 2 / 16.f;
				l.input_layer->weights[i + 8] = 1 / 16.f;
			}
		}
		for (i = 0; i < n; ++i) l.input_layer->biases[i] = 0;
		#ifdef GPU
		if (gpu_index >= 0) {
			l.input_antialiasing_gpu = cuda_make_array(NULL, l.batch*l.outputs);
			push_convolutional_layer(*(l.input_layer));
		}
		#endif  // GPU
	}

	return l;
}
