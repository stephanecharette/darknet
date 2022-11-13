// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include <map>
#include "darknet-ng.hpp"
#include "parser.hpp"


Darknet_ng::ELearningRatePolicy Darknet_ng::learning_rate_policy_from_string(const std::string & str)
{
	const auto name = lowercase(str);

	for (int i = 0; i < static_cast<int>(ELearningRatePolicy::kMax); i ++)
	{
		ELearningRatePolicy learning_rate_policy = static_cast<ELearningRatePolicy>(i);
		const auto str = to_string(learning_rate_policy);
		if (name == str)
		{
			return learning_rate_policy;
		}
	}

	throw std::invalid_argument("invalid learning rate policy: \"" + name + "\"");
}


std::string Darknet_ng::to_string(const Darknet_ng::ELearningRatePolicy & learning_rate_policy)
{
	switch (learning_rate_policy)
	{
		case Darknet_ng::ELearningRatePolicy::kConstant:	return "constant";
		case Darknet_ng::ELearningRatePolicy::kStep:		return "step";
		case Darknet_ng::ELearningRatePolicy::kEXP:			return "exp";
		case Darknet_ng::ELearningRatePolicy::kPoly:		return "poly";
		case Darknet_ng::ELearningRatePolicy::kSteps:		return "steps";
		case Darknet_ng::ELearningRatePolicy::kSigmoid:		return "sigmoid";
		case Darknet_ng::ELearningRatePolicy::kRandom:		return "random";
		case Darknet_ng::ELearningRatePolicy::kSGDR:		return "sgdr";
		case Darknet_ng::ELearningRatePolicy::kMax:			break;
	}

	throw std::invalid_argument("unknown learning rate policy: " + std::to_string(static_cast<int>(learning_rate_policy)));
}
