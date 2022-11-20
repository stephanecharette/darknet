// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


Darknet_ng::Network & Darknet_ng::Network::parse_net(const Section & net)
{
	/// @todo Now that we're passing in the specific config section, rename "net" to "section" so it is consistent with the other parse methods

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

	settings.w								= net.i("width"							, 0);
	settings.h								= net.i("height"						, 0);
	settings.c								= net.i("channels"						, 0);

	if (settings.w < 1 or settings.h < 1 or settings.c < 1)
	{
		throw std::runtime_error("invalid channel or image dimensions in network section");
	}

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

	settings.label_smooth_eps				= net.f("label_smooth_eps"				, 0.0f	);
	settings.resize_step					= net.i("resize_step"					, 32	);
	settings.attention						= net.i("attention"						, 0		);
	settings.adversarial_lr					= net.f("adversarial_lr"				, 0.0f	);
	settings.max_chart_loss					= net.f("max_chart_loss"				, 20.0f	);
	settings.angle							= net.f("angle"							, 0.0f	);
	settings.aspect							= net.f("aspect"						, 1.0f	);
	settings.saturation						= net.f("saturation"					, 1.0f	);
	settings.exposure						= net.f("exposure"						, 1.0f	);
	settings.hue							= net.f("hue"							, 0.0f	);
	settings.power							= net.f("power"							, 4.0f	);

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
