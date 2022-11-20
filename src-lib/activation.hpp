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
}
