// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"
#include "config.hpp"
#include <iostream>
#include <regex>


Darknet_ng::Section::~Section()
{
	return;
}


Darknet_ng::Section::Section(const std::string & n) :
	name(n)
{
	return;
}


#ifdef STEPHANE
Darknet_ng::VStr Darknet_ng::read_cfg(const std::filesystem::path & cfg_filename)
{
	auto v = read_text_file(cfg_filename);

	/* Everything in the .cfg file is one of the following:
	 *
	 *	1) blank lines
	 *	2) comments (starts with "#" or ";")
	 *	3) sections names, such as "[net]"
	 *	4) key-value pairs, such as "batch = 1"
	 */
	const std::regex rx(
		"[#;].*"				// comments we need to ignore
		"|"
		"\\[\\s*(\\S+)\\s*\\]"	// group #1:  new section, such as "[net]"
		"|"						// ...or...
		"(\\S+)"				// group #2:  key
		"\\s*=\\s*"				// =
		"(.*)"					// group #3:  optional value
		);

	MStr options;

	size_t line_number = 0;
	for (auto & line : v)
	{
		line_number ++;
		strip_text(line);
		if (line.empty())
		{
			continue;
		}

		std::smatch m;
		const bool found = std::regex_match(line, m, rx);
		if (not found)
		{
			throw std::runtime_error("failed to parse line #" + std::to_string(line_number) + " in " + cfg_filename.string());
		}

		const std::string section_name	= m.str(1);
		const std::string key			= m.str(2);
		const std::string val			= m.str(3);

		if (not section_name.empty())
		{
			// new section found
			std::cout << "[" << section_name << "]" << std::endl;
		}
		else if (not key.empty())
		{
			// new key-value found
			options[key] = val;
			std::cout << key << "=" << val << std::endl;
		}
	}

	return v;

#ifdef STEPHANE
	// need to process this line
	read_option(line, current->options);

	FILE *file = fopen(filename, "r");
	if(file == 0) file_error(filename);
	char *line;
	int nu = 0;
	list *sections = make_list();
	section *current = 0;
	while((line=fgetl(file)) != 0){
		++ nu;
		strip(line);
		switch(line[0]){
			case '[':
				current = (section*)xmalloc(sizeof(section));
				list_insert(sections, current);
				current->options = make_list();
				current->type = line;
				break;
			case '\0':
			case '#':
			case ';':
				free(line);
				break;
			default:
				if(!read_option(line, current->options)){
					fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
					free(line);
				}
				break;
		}
	}
	fclose(file);
	return sections;
#endif
}
#endif


Darknet_ng::Config::~Config()
{
	return;
}


Darknet_ng::Config::Config()
{
	return;
}


Darknet_ng::Config::Config(const std::filesystem::path & cfg_filename)
{
	read(cfg_filename);

	return;
}


Darknet_ng::Config & Darknet_ng::Config::clear()
{
	sections.clear();

	return *this;
}


bool Darknet_ng::Config::empty() const
{
	return sections.empty();
}


Darknet_ng::Config & Darknet_ng::Config::read(const std::filesystem::path & cfg_filename)
{
	clear();

	auto v = read_text_file(cfg_filename);

	/* Everything in the .cfg file is one of the following:
	 *
	 *	1) blank lines
	 *	2) comments (starts with "#" or ";")
	 *	3) sections names, such as "[net]"
	 *	4) key-value pairs, such as "batch = 1"
	 */
	const std::regex rx(
		"[#;].*"				// comments we need to ignore
		"|"
		"\\[\\s*(\\S+)\\s*\\]"	// group #1:  new section, such as "[net]"
		"|"						// ...or...
		"(\\S+)"				// group #2:  key
		"\\s*=\\s*"				// =
		"(.*)"					// group #3:  optional value
	);

	std::string most_recent_section_name;

	size_t line_number = 0;
	for (auto & line : v)
	{
		line_number ++;
		strip_text(line);
		if (line.empty())
		{
			// ignore blank lines
			continue;
		}

		std::smatch m;
		const bool found = std::regex_match(line, m, rx);
		if (not found)
		{
			throw std::runtime_error("failed to parse line #" + std::to_string(line_number) + " in " + cfg_filename.string());
		}

		const std::string section_name	= m.str(1);
		const std::string key			= m.str(2);
		const std::string val			= m.str(3);

		if (section_name.empty() and key.empty())
		{
			// ignore comments
			continue;
		}

		if (not section_name.empty())
		{
			// new section found
			sections.push_back(section_name);
			continue;
		}

		// we have a key-value pair...but make sure we at least have a section before we attempt to add a new key-value pair

		if (sections.empty())
		{
			throw std::runtime_error("config cannot have values prior to \"[...]\" section name at line " + std::to_string(line_number));
		}

		// add this key-pair to the *most recent* section that we created
		Section & section = *sections.rbegin();
		if (section.kv_pairs.count(key) > 0)
		{
			throw std::runtime_error("[" + section.name + "] already contains " + key + "=" + section.kv_pairs[key] + ", but duplicate key found on line #" + std::to_string(line_number));
		}
		section.kv_pairs[key] = val;
	}

	return *this;
}


std::ostream & Darknet_ng::operator<<(std::ostream & os, const Darknet_ng::Config & cfg)
{
	for (const auto & s : cfg.sections)
	{
		os << "[" << s.name << "]" << std::endl;
		for (const auto & [key, val] : s.kv_pairs)
		{
			std::cout << key << "=" << val << std::endl;
		}
	}

	return os;
}
