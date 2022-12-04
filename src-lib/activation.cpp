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
