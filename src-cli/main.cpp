// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include <iostream>


int main(int argc, char ** argv)
{
	std::cout << "Darknet Next Generation v" << Darknet_ng::version() << std::endl;

#if 0
	Darknet_ng::Config cfg("test.cfg");
	std::cout << cfg << std::endl;

	std::cout
		<< "width ........... " << cfg["net"]["width"]			<< std::endl
		<< "height .......... " << cfg["net"]["height"]			<< std::endl
		<< "learning_rate ... " << cfg["net"]["learning_rate"]	<< std::endl
		<< "yolo layers ..... " << cfg.count("yolo")			<< std::endl
		<< "yolo line # ..... " << cfg["yolo"].line_number		<< std::endl
		<< cfg["net"]											<< std::endl;
#endif

	Darknet_ng::Network network("test.cfg");

	return 0;
}
