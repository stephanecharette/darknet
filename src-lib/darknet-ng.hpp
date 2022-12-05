// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


/// The Darknet-NG namespace.
namespace Darknet_ng
{
	/// @todo get rid of this soon
	#define xcalloc(m, s) calloc(m, s)

	/// @{ Simple type names used throughout Darknet-NG (vectors and maps of commonly-used types).
	using MStr	= std::map		<std::string, std::string	>;
	using VStr	= std::vector	<std::string				>;
	using VI	= std::vector	<int						>;
	using VF	= std::vector	<float						>;
	/// @}

	/// Get the version string.  Looks like @p "1.2.3-1".
	std::string version();

	/// @{ Strip leading and trailing whitespace from the line of text.
	std::string strip_text(const std::string & line);
	std::string & strip_text(std::string & line);
	/// @}

	/// Get the lowercase version of the given text.  The original string is untouched.
	std::string lowercase(const std::string & text);

	/// Convert the given string to lowercase.
	std::string & lowercase(std::string & text);

	/// Read the given text file line-by-line and store in a vector.  File must exist.
	VStr read_text_file(const std::filesystem::path & filename);

	/// Generate a random float.
	float rand_uniform(float low, float high);
}


#include "Exception.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "enums.hpp"
#include "structs.hpp"
#include "Activation.hpp"
#include "LearningRatePolicy.hpp"
#include "Layers.hpp"
#include "Config.hpp"
#include "Network.hpp"
