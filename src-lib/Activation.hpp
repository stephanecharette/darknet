// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <string>


namespace Darknet_ng
{
	enum class EActivation /// was: ACTIVATION
	{
		kLogistic				,
		kRELU					,
		kRELU6					,
		kRELIE					,
		kLinear					,
		kRamp					,
		kTANH					,
		kPLSE					,
		kRevLeaky				,
		kLeaky					,
		kELU					,
		kLOGGY					,
		kStair					,
		kHardTAN				,
		kLHTAN					,
		kSELU					,
		kGELU					,
		kSWISH					,
		kMISH					,
		kHardMISH				,
		kNormCHAN				,
		kNormCHANSoftmax		,
		kNormCHANSoftmaxMaxVal	,
		// remember to update to_string() if you add a new activation
		kMax
	};

	/// Get the activation from the given string.  Must be an exact match, otherwise this will throw.
	EActivation activation_from_string(const std::string & str);

	/// Convert the activation to a text string.
	std::string to_string(const EActivation & activation);

	void activate_array_swish						(float * x, const int n, float * output_sigmoid		, float * output);
	void activate_array_mish						(float * x, const int n, float * activation_input	, float * output);
	void activate_array_hard_mish					(float * x, const int n, float * activation_input	, float * output);
	void activate_array_normalize_channels			(float * x, const int n, int batch, int channels, int wh_step, float * output);
	void activate_array_normalize_channels_softmax	(float * x, const int n, int batch, int channels, int wh_step, float * output, int use_max_val);

	/// @todo This was a static.  Looks like a GPU version also exists?  Should this be exposed?
	float hard_mish_yashas(float x);

	static inline float logistic_activate(const float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}

	static inline float softplus_activate(float x, float threshold)
	{
		if (x > threshold)
		{
			return x;		// too large
		}

		if (x < -threshold)
		{
			return expf(x);	// too small
		}

		return logf(expf(x) + 1);
	}

	static inline float tanh_activate(float x)
	{
		return (2 / (1 + expf(-2 * x)) - 1);
	}

	void activate_array_cpu_custom(float * x, const int n, const EActivation a);
}
