// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include "enums.hpp"
#include "structs.hpp"


namespace Darknet_ng
{
	using VStr = std::vector<std::string>;
	using MStr = std::map<std::string, std::string>;
	using VI = std::vector<int>;
	using VF = std::vector<float>;

	/// Get the version string.  Looks like @p "1.2.3-1".
	std::string version();

	/// @{ Strip leading and trailing whitespace from the line of text.
	std::string strip_text(const std::string & line);
	std::string & strip_text(std::string & line);
	/// @}

	/// Get the lowercase version of the given text.  The original is untouched.
	std::string lowercase(const std::string & text);

	/// Convert the given string to lowercase.
	std::string & lowercase(std::string & text);

	/// Read the given text file.  File must exist.
	VStr read_text_file(const std::filesystem::path & filename);

	/// @{ Load network and get batch size from cfg file.
//	Network *load_network(char *cfg, char *weights, int clear);
//	Network *load_network_custom(char *cfg, char *weights, int clear, int batch);
	Network load_network(
		const std::filesystem::path & cfg_filename,
		const std::filesystem::path & weights_filename,
		const bool clear = true);
	/// @}

	Network parse_network_cfg(const std::filesystem::path & cfg_filename, int batch = 0, int time_steps = 0);
	Network parse_network_cfg_custom(const std::filesystem::path & cfg_filename, int batch, int time_steps = 0);
}


#include "config.hpp"
