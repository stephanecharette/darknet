// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include <algorithm>
#include <fstream>
#include <random>
#include <unistd.h>
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
		/// @throw Exception The file does not exist.
		throw Exception("file does not exist: \"" + filename.string() + "\"", DNG_LOC);
	}

	std::ifstream ifs(filename);
	if (not ifs.good())
	{
		/// @throw Exception The file cannot be read.  (Permission issue?)
		throw Exception("failed to read file: \"" + filename.string() + "\"", DNG_LOC);
	}

	VStr v;
	std::string line;
	while (std::getline(ifs, line))
	{
		v.push_back(line);
	}

	return v;
}


std::default_random_engine & get_engine()
{
	static std::default_random_engine engine(
		[]() -> uint64_t
		{
			const auto now		= std::chrono::system_clock::now();
			const auto duration	= now.time_since_epoch();
			const uint64_t ns	= std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
			const uint64_t pid	= getpid();
			const uint64_t seed	= pid * ns;
std::cout << "SEED=" << seed << std::endl; ///< @todo remove this line
			return seed;
		}()
	);

	return engine;
}


float Darknet_ng::rand_uniform(float low, float high)
{
	if (low > high)
	{
		std::swap(low, high);
	}

	/** @todo Note that the range is [low, high) meaning if you pass in 0.0f and 1.0f, you might get 0.999999999f but
	 * never 1.0f.  Need to look up what the original function did to see if this behaviour is similar, and possibly
	 * also where it is used to see if we should care.
	 */
	std::uniform_real_distribution<float> distribution(low, high);

	auto result = distribution(get_engine());

std::cout << "RND LOW=" << low << " HIGH=" << high << " RESULT=" << result << std::endl; /// @todo remove this line

	return result;
}
