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

	Darknet_ng::Network network;
	network.load("test.cfg");

	std::cout
		<< "gpu_index="				<< network.settings.gpu_index			<< std::endl
		<< "max_batches="			<< network.settings.max_batches			<< std::endl
		<< "batch="					<< network.settings.batch				<< std::endl
		<< "learning_rate="			<< network.settings.learning_rate		<< std::endl
		<< "learning_rate_min="		<< network.settings.learning_rate_min	<< std::endl
		<< "batches_per_cycle="		<< network.settings.batches_per_cycle	<< std::endl
		<< "batches_cycle_mult="	<< network.settings.batches_cycle_mult	<< std::endl
		<< "momentum="				<< network.settings.batches_cycle_mult	<< std::endl
		<< "decay="					<< network.settings.decay				<< std::endl
		<< "subdivisions="			<< network.settings.subdivisions		<< std::endl
		<< "EMPTY="					<< network.empty()						<< std::endl
		<< "LOADED="				<< network.loaded()						<< std::endl;

	network.clear();

	std::cout
		<< "gpu_index="				<< network.settings.gpu_index			<< std::endl
		<< "max_batches="			<< network.settings.max_batches			<< std::endl
		<< "batch="					<< network.settings.batch				<< std::endl
		<< "learning_rate="			<< network.settings.learning_rate		<< std::endl
		<< "learning_rate_min="		<< network.settings.learning_rate_min	<< std::endl
		<< "batches_per_cycle="		<< network.settings.batches_per_cycle	<< std::endl
		<< "batches_cycle_mult="	<< network.settings.batches_cycle_mult	<< std::endl
		<< "momentum="				<< network.settings.batches_cycle_mult	<< std::endl
		<< "decay="					<< network.settings.decay				<< std::endl
		<< "subdivisions="			<< network.settings.subdivisions		<< std::endl
		<< "EMPTY="					<< network.empty()						<< std::endl
		<< "LOADED="				<< network.loaded()						<< std::endl;

	return 0;
}
