// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


namespace Darknet_ng
{
	/** A "section" is the [...] name and all key-value pairs that follow that name.
	 * Note that sections are not unique.  For example, a configuration file might
	 * have multiple [YOLO] sections, so don't store these in a set or a map.
	 */
	class Section final
	{
		public:

			/// Destructor.
			~Section();

			/// Constructor.
			Section(const std::string & n);

			/// The section names are enclosed in square brackets, such as @p "[net]".
			std::string name;

			/** Every section is composed of zero or more key-value pairs, such as @p "classes = 12".
			 * The exact same key name cannot appear multiple times within a section.
			 */
			MStr kv_pairs;
	};
	using Sections = std::vector<Section>;

	/** The configuration file is a series of sections and options.
	 * This class does not maintain comments and blank lines.
	 */
	class Config final
	{
		public:

			/// Destructor.
			~Config();

			/// Empty constructor.  Call @ref read() afterwards.
			Config();

			/// Constructor that automatically calls @ref read().
			Config(const std::filesystem::path & cfg_filename);

			/// Drop any previous configuration.
			Config & clear();

			/// Returns @p true if no configuration has been read, or if @ref clear() has been called.
			bool empty() const;

			/// Read in the given configuration file.  Skips over comments and blank lines.
			Config & read(const std::filesystem::path & cfg_filename);

			/** All the section names and key-value pairs stored in a configuration.
			 * Note that comments and blank lines are removed.
			 */
			Sections sections;
	};

	/// Convenience function to stream a configuration as plain text.
	std::ostream & operator<<(std::ostream & os, const Config & cfg);
}
