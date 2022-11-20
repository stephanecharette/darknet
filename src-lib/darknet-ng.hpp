// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>


/// The Darknet-NG namespace.
namespace Darknet_ng
{
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
}


#include "enums.hpp"
#include "structs.hpp"
#include "activation.hpp"
#include "parser.hpp"
#include "layers.hpp"
#include "config.hpp"
#include "network.hpp"
