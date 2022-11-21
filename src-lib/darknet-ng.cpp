// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include <algorithm>
#include <fstream>
#include <random>
#include "darknet-ng.hpp"


std::string Darknet_ng::version()
{
	return DNG_VERSION;
}


std::string Darknet_ng::strip_text(const std::string & line)
{
	std::string txt = line;

	return strip_text(txt);
}


std::string & Darknet_ng::strip_text(std::string & line)
{
	if (not line.empty())
	{
		// trailing whitespace
		auto pos = line.find_last_not_of(" \t\r\n");
		if (pos != std::string::npos)
		{
			line.erase(pos + 1);
		}

		// leading whitespace
		pos = line.find_first_not_of(" \t\r\n");
		if (pos != std::string::npos and pos > 0)
		{
			line.erase(0, pos);
		}
	}

	return line;
}


std::string Darknet_ng::lowercase(const std::string & text)
{
	std::string str = text;

	return lowercase(str);
}


std::string & Darknet_ng::lowercase(std::string & text)
{
	std::transform(text.begin(), text.end(), text.begin(),
		[](unsigned char c)
		{
			return std::tolower(c);
		});

	return text;
}


Darknet_ng::VStr Darknet_ng::read_text_file(const std::filesystem::path & filename)
{
	if (not std::filesystem::exists(filename))
	{
		throw std::invalid_argument("file does not exist: \"" + filename.string() + "\"");
	}

	std::ifstream ifs(filename);
	if (not ifs.good())
	{
		throw std::invalid_argument("failed to read file: \"" + filename.string() + "\"");
	}

	VStr v;
	std::string line;
	while (std::getline(ifs, line))
	{
		v.push_back(line);
	}

	return v;
}


float Darknet_ng::rand_uniform(float low, float high)
{
	if (low < high)
	{
		std::swap(low, high);
	}

	static std::default_random_engine engine;
	static std::uniform_real_distribution<> distribution(low, high); // rage 0 - 1

	return distribution(engine);
}
