// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <string>


namespace Darknet_ng
{
	/// @todo
	enum UNUSED_ENUM_TYPE
	{
		UNUSED_DEF_VAL
	};

	/// was:  LAYER_TYPE
	enum class ELayerType
	{
		kConvolutional		,
		kDeconvolutional	,
		kConnected			,
		kMaxPool			,
		kLocalAvgPool		,
		kSoftMax			,
		kDetection			,
		kDropout			,
		kCrop				,
		kRoute				,
		kCost				,
		kNormalization		,
		kAvgPool			,
		kLocal				,
		kShortcut			,
		kScaleChannels		,
		kSam				,
		kActive				,
		kRNN				,
		kGRU				,
		kLSTM				,
		kConvLSTM			,
		kHistory			,
		kCRNN				,
		kBatchNorm			,
		kNetwork			,
		kXNOR				,
		kRegion				,
		kYOLO				,
		kGaussianYOLO		,
		kISEG				,
		kReorg				,
		kReorgOld			,
		kUpsample			,
		kLogXent			,
		kL2Norm				,
		kEmpty				,
		kBlank				,
		kContrastive		,
		kImplicit
	};

	enum class ECostType
	{
		kSSE	,
		kMasked	,
		kL1		,
		kSeg	,
		kSmooth	,
		kWGAN
	};

	enum class EIOULoss // was: IOU_LOSS
	{
		kIOU	,
		kGIOU	,
		kMSE	,
		kDIOU	,
		kCIOU
	};

	enum class ENMSKind // was: NMS_KIND
	{
		kDefaultNMS	,
		kGreedyNMS	,
		kDIOU_NMS	,
		kCornersNMS
	};

	enum class EYOLOPoint // was: YOLO_POINT
	{
		kYOLOCenter			= 0x01,
		kYOLOLeftTop		= 0x02,
		kYOLORightBottom	= 0x04
	};

	enum class EWeightsType // was: WEIGHTS_TYPE_T
	{
		kNoWeights	,
		kPerFeature	,
		kPerChannel
	};

	enum class EWeightsNormalization // was: WEIGHTS_NORMALIZATION_T
	{
		kNoNormalization	,
		kRELUNormalization	,
		kSoftmaxNormalization
	};

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
		kNormCHANSoftmaxMaxVal
	};

	enum class EBinaryActivation /// was: BINARY_ACTIVATION
	{
		kMult	,
		kAdd	,
		kSub	,
		kDiv
	};

	enum class ELearningRatePolicy // was:  learning_rate_policy
	{
		kConstant	,
		kStep		,
		kEXP		,
		kPoly		,
		kSteps		,
		kSIG		,
		kRandom		,
		kSGDR
	};

	enum class EDataType // was: data_type
	{
		kClassificationData		,
		kDetectionData			,
		kCaptchaData			,
		kRegionData				,
		kImageData				,
		kCompareData			,
		kWritingData			,
		kSwagData				,
		kTagData				,
		kOldClassificationData	,
		kStudyData				,
		kDETData				,
		kSuperData				,
		kLetterboxData			,
		kRegressionData			,
		kSegmentationData		,
		kInstanceData			,
		kISEGData
	};

	enum class EImType // was: IMTYPE
	{
		kPNG	,
		kBMP	,
		kTGA	,
		kJPG
	};
}
