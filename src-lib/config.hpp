// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


namespace Darknet_ng
{
	/** A "section" is the [...] name and all key-value pairs that follow that name.  Note that sections are not unique.
	 * For example, a configuration file might have multiple @p [yolo] sections, so don't store these in a set or a map.
	 *
	 * Sections names and keys are automatically converted to lowercase when a configuration file is parsed.  An example
	 * section might be:
	 *
	 * ~~~~{.txt}
	 * [maxpool]
	 * size=2
	 * stride=2
	 * ~~~~
	 *
	 * In this case the section name is @p maxpool and the two key-value pairs are:
	 *
	 * @li key = @p size and val = @p 2
	 * @li key = @p stride and val = @p 2
	 *
	 * @warning Keys are stored alphabetically in @ref kv_pairs, not in the order in which they have been added to the section.
	 * This means if you parse a configuration file and then output the sections, the results may not necessarily match.
	 *
	 * @see @ref Darknet_ng::Config
	 */
	class Section final
	{
		public:

			/// Destructor.
			~Section();

			/// Constructor.
			Section(const std::string & n);

			/// Constructor.
			Section(const std::string & n, const size_t line);

			/** This return @p true only when @p kv_pairs is empty.  As soon as the section has a key-value pair then we
			 * consider it as non-empty.
			 */
			bool empty() const;

			/// This will remove the name, line number, and any key-value pairs.
			Section & clear();

			/// Compare two @p Section objects.
			bool operator==(const Section & rhs) const;

			/// Get the value for a specific key.  The key-value pair must exist, or this will throw an exception.
			const std::string & operator[](const std::string & key) const;

			/// Get the value corresponding to a key, and convert it to an integer.  The key must exist, and must be numeric.
			int i(const std::string & key) const;

			/// Get the value corresponding to a key, and convert it to an integer.  If the key does not exist, use the provided default value.
			int i(const std::string & key, const int default_value) const;

			/// Get the value corresponding to a key, and convert it to a float.  The key must exist, and must be numeric.
			float f(const std::string & key) const;

			/// Get the value corresponding to a key, and convert it to a float.  If the key does not exist, use the provided default value.
			float f(const std::string & key, const float default_value) const;

			/// Get the value corresponding to a key, and convert it to a bool.  The key must exist, and must be @p 0, @p 1, @p true, or @p false.
			bool b(const std::string & key) const;

			/// Get the value corresponding to a key, and convert it to a bool.  If the key does not exist, use the provided default value.
			bool b(const std::string & key, const bool default_value) const;

			/// The section names are enclosed in square brackets, such as @p "[net]".
			std::string name;

			/// The line number where this section started in the original @p .cfg file.
			size_t line_number;

			/** Every section is composed of zero or more key-value pairs, such as @p "classes = 12" where the key would be
			 * @p classes and the value would be @p 12.  The exact same key name cannot appear multiple times within a section.
			 */
			MStr kv_pairs;
	};
	using Sections = std::vector<Section>;

	/** The configuration file is a series of sections and options.  This class does not maintain comments and blank lines.
	 * When a configuration file is parsed by @ref read(), all section names and keys will be converted to lowercase.
	 *
	 * @warning The sections are stored in the exact order in which they are read, but the individual key-value pairs within
	 * each section may be re-ordered.  See @ref Section for additional details.
	 *
	 * A configuration file example might include:
	 * ~~~~{.txt}
	 * [net]
	 * # Testing
	 * #batch=1
	 * #subdivisions=1
	 * # Training
	 * batch=1
	 * subdivisions=1
	 * width=416
	 * height=416
	 * ~~~~
	 *
	 * This example has 1 section named @p net which contains 4 key-value pairs.  Remember that comments are not retained,
	 * so 4 of the lines from this example are ignored.
	 *
	 * To display the width, height, and several other configuration settings for this neural network, you could use:
	 *
	 * ~~~~
	 * Darknet_ng::Config cfg("test.cfg");
	 * std::cout
	 *		<< "width ......... " << cfg["net"]["width"]     << std::endl
	 *		<< "height ........ " << cfg["net"]["height"]    << std::endl
	 *		<< "yolo layers ... " << cfg.count("yolo")       << std::endl
	 *		<< "yolo line # ... " << cfg["yolo"].line_number << std::endl
	 *		<< cfg["yolo"]                                   << std::endl;
	 * ~~~~
	 *
	 * @see @ref Darknet_ng::Section
	 */
	class Config final
	{
		public:

			/// Destructor.
			~Config();

			/// Empty constructor.  Try calling @ref read() afterwards to import the configuration.
			Config();

			/// Constructor that automatically calls @ref read().
			Config(const std::filesystem::path & cfg_filename);

			/// Drop any previous configuration.
			Config & clear();

			/// Returns @p true if no configuration has been read, or if @ref clear() has been called.
			bool empty() const;

			/** Parse the given configuration file.  Skips over comments and blank lines.  Each section of the configuration file
			 * will be stored in @ref sections.
			 * @see @ref find()
			 * @see @ref find_next()
			 * @see @ref operator[]()
			 */
			Config & read(const std::filesystem::path & cfg_filename);

			/// Count the number of sections with the given name.
			size_t count(const std::string & name) const;

			/** Find the first section that matches this name.
			 * Section name must exist, otherwise an exception will be thrown.
			 */
			const Section & find(const std::string & name) const;

			/** Alias for @ref find().  You can do things like:
			 * ~~~~
			 * Darknet_ng::Config cfg("test.cfg");
			 * std::cout << "width=" << cfg["net"]["width"] << std::endl;
			 * ~~~~
			 */
			const Section & operator[](const std::string & name) { return find(name); }

			/** Similar to @ref find() but starts searching after the given section.
			 * Section name must exist, otherwise an exception will be thrown.
			 */
			const Section & find_next(const std::string & name, const Section & previous_section) const;

			/** All the section names and key-value pairs stored in a configuration.
			 * Note that comments and blank lines are removed.
			 */
			Sections sections;
	};

	/** Convenience function to stream a @ref Section as plain text.  This is mostly for debug purposes.  Remember that
	 * the keys in a section are stored alphabetically, so the order produced by calling this may not be exactly the same
	 * as the original configuration file.
	 */
	std::ostream & operator<<(std::ostream & os, const Section & section);

	/** Convenience function to stream a configuration as plain text.  This is mostly for debug purposes.  Remember that
	 * @ref Sections store keys alphabetically, so the order produced by calling this may not be exactly the same as the
	 * original configuration file that was parsed by @ref Config::read().
	 */
	std::ostream & operator<<(std::ostream & os, const Config & cfg);
}
