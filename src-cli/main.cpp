// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include <iostream>


int main(int argc, char ** argv)
{
	std::cout << "Darknet Next Generation v" << Darknet_ng::version() << std::endl;

	Darknet_ng::Config cfg("test.cfg");
	std::cout << cfg << std::endl
		<< "Sections ... " << cfg.sections.size() << std::endl;


//	Darknet_ng::VStr v = Darknet_ng::read_cfg("test.cfg");

	return 0;
}
