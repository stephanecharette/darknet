// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


Darknet_ng::EActivation Darknet_ng::activation_from_string(const std::string & str)
{
	const auto name = lowercase(str);

	for (int i = 0; i < static_cast<int>(EActivation::kMax); i ++)
	{
		EActivation activation = static_cast<EActivation>(i);
		const auto str = to_string(activation);
		if (name == str)
		{
			return activation;
		}
	}

	/// @throw Exception The string provided must be an activation name.
	throw Exception("invalid activation: \"" + name + "\"", DNG_LOC);
}


std::string Darknet_ng::to_string(const Darknet_ng::EActivation & activation)
{
	switch (activation)
	{
		case EActivation::kLogistic:				return "logistic";
		case EActivation::kRELU:					return "relu";	// this was the default in the original darknet if an activation string didn't match
		case EActivation::kRELU6:					return "relu6";
		case EActivation::kRELIE:					return "relie";
		case EActivation::kLinear:					return "linear";
		case EActivation::kRamp:					return "ramp";
		case EActivation::kTANH:					return "tanh";
		case EActivation::kPLSE:					return "plse";
		case EActivation::kRevLeaky:				return "revleaky";
		case EActivation::kLeaky:					return "leaky";
		case EActivation::kELU:						return "elu";
		case EActivation::kLOGGY:					return "loggy";
		case EActivation::kStair:					return "stair";
		case EActivation::kHardTAN:					return "hardtan";
		case EActivation::kLHTAN:					return "lhtan";
		case EActivation::kSELU:					return "selu";
		case EActivation::kGELU:					return "gelu";
		case EActivation::kSWISH:					return "swish";
		case EActivation::kMISH:					return "mish";
		case EActivation::kHardMISH:				return "hard_mish";
		case EActivation::kNormCHAN:				return "normalize_channels";
		case EActivation::kNormCHANSoftmax:			return "normalize_channels_softmax";
		case EActivation::kNormCHANSoftmaxMaxVal:	return "normalize_channels_softmax_maxval";
		case EActivation::kMax:						break;
	}

	/// @throw Exception The activation enum is unknown or not supported.
	throw Exception("unknown or invalid activation enum: " + std::to_string(static_cast<int>(activation)), DNG_LOC);
}


void Darknet_ng::activate_array_swish(float *x, const int n, float * output_sigmoid, float * output)
{
	int i;
	#pragma omp parallel for
	for (i = 0; i < n; ++i)
	{
		float x_val = x[i];
		float sigmoid = logistic_activate(x_val);
		output_sigmoid[i] = sigmoid;
		output[i] = x_val * sigmoid;
	}

	return;
}


void Darknet_ng::activate_array_mish(float * x, const int n, float * activation_input, float * output)
{
	// https://github.com/digantamisra98/Mish

	const float MISH_THRESHOLD = 20.0f;

	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		float x_val = x[i];
		activation_input[i] = x_val;    // store value before activation
		output[i] = x_val * tanh_activate( softplus_activate(x_val, MISH_THRESHOLD) );
	}
}


void Darknet_ng::activate_array_hard_mish(float * x, const int n, float * activation_input, float * output)
{
	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		float x_val = x[i];
		activation_input[i] = x_val;    // store value before activation
		output[i] = hard_mish_yashas(x_val);
	}

	return;
}


void Darknet_ng::activate_array_normalize_channels(float * x, const int n, int batch, int channels, int wh_step, float * output)
{
	int size = n / channels;

	#pragma omp parallel for
	for (int i = 0; i < size; ++i)
	{
		int wh_i = i % wh_step;
		int b = i / wh_step;

		const float eps = 0.0001;
		if (i < size)
		{
			float sum = eps;
			int k;
			for (k = 0; k < channels; ++k)
			{
				float val = x[wh_i + k * wh_step + b * wh_step * channels];
				if (val > 0)
				{
					sum += val;
				}
			}
			for (k = 0; k < channels; ++k)
			{
				float val = x[wh_i + k * wh_step + b * wh_step * channels];
				if (val > 0)
				{
					val = val / sum;
				}
				else
				{
					val = 0;
				}
				output[wh_i + k * wh_step + b*wh_step*channels] = val;
			}
		}
	}

	return;
}


void Darknet_ng::activate_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *output, int use_max_val)
{
	int size = n / channels;

	#pragma omp parallel for
	for (int i = 0; i < size; ++i)
	{
		int wh_i = i % wh_step;
		int b = i / wh_step;

		const float eps = 0.0001;
		if (i < size)
		{
			float sum = eps;
			float max_val = -FLT_MAX;
			if (use_max_val)
			{
				for (int k = 0; k < channels; ++k)
				{
					float val = x[wh_i + k * wh_step + b * wh_step * channels];
					if (val > max_val || k == 0)
					{
						max_val = val;
					}
				}
			}
			else
			{
				max_val = 0;
			}

			for (int k = 0; k < channels; ++k)
			{
				float val = x[wh_i + k * wh_step + b * wh_step * channels];
				sum += expf(val - max_val);
			}
			for (int k = 0; k < channels; ++k)
			{
				float val = x[wh_i + k * wh_step + b * wh_step * channels];
				val = expf(val - max_val) / sum;
				output[wh_i + k * wh_step + b * wh_step * channels] = val;
			}
		}
	}

	return;
}


/*static*/ float Darknet_ng::hard_mish_yashas(float x)
{
	if (x > 0.0f)
	{
		return x;
	}
	if (x > -2.0f)
	{
		return x * x / 2.0f + x;
	}

	return 0;
}
