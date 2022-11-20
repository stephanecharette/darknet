// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


const Darknet_ng::MStrLayerType & Darknet_ng::get_all_layer_types_map()
{
	static MStrLayerType m;

	if (m.empty())
	{
		// build a "cache" of all the names and layer types which we'll store in a map:
		//
		//	- the "key" is the text string, such as "yolo"
		//	- the "val" is the layer type, such as ELayerType::kYOLO
		//
		for (int i = 0; i < static_cast<int>(ELayerType::kMax); i ++)
		{
			// several layer types don't have names (!?) and will throw when we query for the name
			try
			{
				ELayerType layer_type = static_cast<ELayerType>(i);
				const auto str = to_string(layer_type);
				m[str] = layer_type;
			}
			catch (...)
			{
				// ignore the layers that don't have names
			}
		}

		// some layer types have an older name that may still be in use in older config files;
		// these are "aliases" for existing layers
		m["conv"		] = ELayerType::kConvolutional;
		m["network"		] = ELayerType::kNetwork;
		m["conn"		] = ELayerType::kConnected;
		m["max"			] = ELayerType::kMaxPool;
		m["local_avg"	] = ELayerType::kLocalAvgPool;
		m["avg"			] = ELayerType::kAvgPool;
		m["lrn"			] = ELayerType::kNormalization; /// @todo this is possibly unused?
		m["soft"		] = ELayerType::kSoftMax;
		m["silence"		] = ELayerType::kEmpty;
	}

	return m;
}


Darknet_ng::ELayerType Darknet_ng::layer_type_from_string(const std::string & str)
{
	const auto name = lowercase(str);

	const auto & m = get_all_layer_types_map();
	if (m.count(name) == 1)
	{
		return m.at(name);
	}

	throw std::invalid_argument("invalid layer type: \"" + name + "\"");
}


std::string Darknet_ng::to_string(const Darknet_ng::ELayerType & layer_type)
{
	switch (layer_type)
	{
		case ELayerType::kConvolutional:	return "convolutional"; // also see "conv"
		case ELayerType::kConnected:		return "connected";
		case ELayerType::kMaxPool:			return "maxpool";
		case ELayerType::kLocalAvgPool:		return "local_avgpool";
		case ELayerType::kSoftMax:			return "softmax";
		case ELayerType::kDetection:		return "detection";
		case ELayerType::kDropout:			return "dropout";
		case ELayerType::kCrop:				return "crop";
		case ELayerType::kRoute:			return "route";
		case ELayerType::kCost:				return "cost";
		case ELayerType::kNormalization:	return "normalization";	///< @todo this is possibly unused?
		case ELayerType::kAvgPool:			return "avgpool";
		case ELayerType::kLocal:			return "local";
		case ELayerType::kShortcut:			return "shortcut";
		case ELayerType::kScaleChannels:	return "scale_channels";
		case ELayerType::kSam:				return "sam";
		case ELayerType::kActive:			return "activation";
		case ELayerType::kRNN:				return "rnn";
		case ELayerType::kGRU:				return "gru";
		case ELayerType::kLSTM:				return "lstm";
		case ELayerType::kConvLSTM:			return "conv_lstm";
		case ELayerType::kHistory:			return "history";
		case ELayerType::kCRNN:				return "crnn";
		case ELayerType::kBatchNorm:		return "batchnorm";
		case ELayerType::kNetwork:			return "net";
		case ELayerType::kRegion:			return "region";
		case ELayerType::kYOLO:				return "yolo";
		case ELayerType::kGaussianYOLO:		return "gaussian_yolo";
		case ELayerType::kReorg:			return "reorg3d";
		case ELayerType::kReorgOld:			return "reorg";
		case ELayerType::kUpsample:			return "upsample";
		case ELayerType::kEmpty:			return "empty";
		case ELayerType::kContrastive:		return "contrastive";
		case ELayerType::kImplicit:			return "implicit";
		case ELayerType::kDeconvolutional:	break; ///< @todo does this exist?
		case ELayerType::kXNOR:				break; ///< @todo does this exist?
		case ELayerType::kISEG:				break; ///< @todo does this exist?
		case ELayerType::kLogXent:			break; ///< @todo does this exist?
		case ELayerType::kL2Norm:			break; ///< @todo does this exist?
		case ELayerType::kBlank:			break; ///< @todo does this exist?
		case ELayerType::kMax:				break;
	}

	throw std::invalid_argument("unknown layer type: " + std::to_string(static_cast<int>(layer_type)));
}
