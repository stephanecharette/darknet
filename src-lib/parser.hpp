// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <string>


namespace Darknet_ng
{
	/// Was:  learning_rate_policy
	enum class ELearningRatePolicy
	{
		kConstant	= 0,
		kStep		,
		kEXP		,
		kPoly		,
		kSteps		,
		kSigmoid	,
		kRandom		,
		kSGDR		,
		// remember to update to_string() if you add a new learning rate policy
		kMax
	};

	/// Get the learning rate policy from the given string.  Must be an exact match, otherwise this will throw.
	ELearningRatePolicy learning_rate_policy_from_string(const std::string & str);

	/// Convert the learning rate policy to a text string.
	std::string to_string(const ELearningRatePolicy & learning_rate_policy);
}
