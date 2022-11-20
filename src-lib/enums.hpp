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


	enum class EBinaryActivation /// was: BINARY_ACTIVATION
	{
		kMult	,
		kAdd	,
		kSub	,
		kDiv
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
