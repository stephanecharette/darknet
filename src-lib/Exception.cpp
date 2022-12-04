// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "Exception.hpp"


Darknet_ng::Exception::Exception(const std::string & msg, const std::string & fn, const std::string & func, const int line) :
	std::runtime_error(msg),
	filename(fn),
	function(func),
	line_number(line)
{
	return;
}


Darknet_ng::Exception::~Exception()
{
	return;
}


const char * Darknet_ng::Exception::what() const noexcept
{
	std::string msg = std::exception::what();

	msg += " [";
	if (not filename.empty())
	{
		msg += filename + ", ";
	}
	if (not function.empty())
	{
		msg += function + ", ";
	}
	msg += "#" + std::to_string(line_number) + "]";

	return msg.c_str();
}
