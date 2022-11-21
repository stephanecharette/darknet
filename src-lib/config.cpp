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
	name(Darknet_ng::lowercase(n)),
	line_number(0)
{
	return;
}


Darknet_ng::Section::Section(const std::string & n, const size_t line) :
	name(Darknet_ng::lowercase(n)),
	line_number(line)
{
	return;
}


bool Darknet_ng::Section::empty() const
{
	return kv_pairs.empty();
}


Darknet_ng::Section & Darknet_ng::Section::clear()
{
	line_number	= 0;
	kv_pairs	.clear();
	name		.clear();

	return *this;
}


bool Darknet_ng::Section::operator==(const Darknet_ng::Section & rhs) const
{
	return
		name		== rhs.name and
		kv_pairs	== rhs.kv_pairs
		/* The line number is meta-data used for debugging.
		 * Don't bother comparing the line number.
		 *
		line_number == rhs.line_number
		 */
		;
}


const std::string & Darknet_ng::Section::operator[](const std::string & key) const
{
	return s(key);
}


int Darknet_ng::Section::i(const std::string & key) const
{
	const auto key_name = lowercase(key);

	const std::string & val = kv_pairs.at(key_name);

	std::cout << "i[" << key_name << "]=" << val << std::endl;

	int result = 0;
	try
	{
		result = std::stod(val);
	}
	catch (std::exception & e)
	{
		throw std::logic_error(key_name + " in section [" + name + "] cannot be converted to an integer (" + val + ")");
	}

	return result;
}


int Darknet_ng::Section::i(const std::string & key, const int default_value) const
{
	const auto key_name = lowercase(key);

	if (kv_pairs.count(key_name) == 0)
	{
		std::cout << "i[" << key_name << "]=" << default_value << " (default value)" << std::endl;
		return default_value;
	}

	return i(key_name);
}


float Darknet_ng::Section::f(const std::string & key) const
{
	const auto key_name = lowercase(key);

	const std::string & val = kv_pairs.at(key_name);

	std::cout << "f[" << key_name << "]=" << val << std::endl;

	float result = 0.0f;
	try
	{
		result = std::stof(val);
	}
	catch (std::exception & e)
	{
		throw std::logic_error(key_name + " in section [" + name + "] cannot be converted to a float (" + val + ")");
	}

	return result;
}


float Darknet_ng::Section::f(const std::string & key, const float default_value) const
{
	const auto key_name = lowercase(key);

	if (kv_pairs.count(key_name) == 0)
	{
		std::cout << "f[" << key_name << "]=" << default_value << " (default value)" << std::endl;
		return default_value;
	}

	return f(key_name);
}


bool Darknet_ng::Section::b(const std::string & key) const
{
	const auto key_name = lowercase(key);
	const std::string val = lowercase(kv_pairs.at(key_name));

	if (val == "0" or
		val == "f" or
		val == "off" or
		val == "false")
	{
		std::cout << "b[" << key_name << "]=false" << std::endl;
		return false;
	}

	if (val == "1" or
		val == "t" or
		val == "on" or
		val == "true")
	{
		std::cout << "b[" << key_name << "]=true" << std::endl;
		return true;
	}

	throw std::logic_error(key_name + " in section [" + name + "] cannot be converted to a bool (" + val + ")");
}


bool Darknet_ng::Section::b(const std::string & key, const bool default_value) const
{
	const auto key_name = lowercase(key);

	if (kv_pairs.count(key_name) == 0)
	{
		std::cout << "b[" << key_name << "]=" << default_value << " (default value)" << std::endl;
		return default_value;
	}

	return b(key_name);
}


const std::string & Darknet_ng::Section::s(const std::string & key) const
{
	const auto key_name = lowercase(key);

	std::cout << "f[" << key_name << "]=" << kv_pairs.at(key_name) << std::endl;

	return kv_pairs.at(key_name);
}


std::string Darknet_ng::Section::s(const std::string & key, const std::string & default_value) const
{
	const auto key_name = lowercase(key);

	if (kv_pairs.count(key_name) == 0)
	{
		std::cout << "s[" << key_name << "]=" << default_value << " (default value)" << std::endl;
		return default_value;
	}

	return kv_pairs.at(key_name);
}


Darknet_ng::VI Darknet_ng::Section::vi(const std::string & key) const
{
	const auto key_name = lowercase(key);

	// for example:  steps=-1,100,20000,30000

	VI v;
	std::string token;
	std::stringstream ss(s(key_name, ""));
	std::cout << "vi[" << key_name << "]=";

	while (std::getline(ss, token, ','))
	{
		const auto val = std::stoi(token);
		std::cout << val << " ";
		v.push_back(val);
	}
	std::cout << std::endl;

	return v;
}


Darknet_ng::VF Darknet_ng::Section::vf(const std::string & key) const
{
	const auto key_name = lowercase(key);

	// for example:  scales=.1,.1

	VF v;
	std::string token;
	std::stringstream ss(s(key_name, ""));
	std::cout << "vf[" << key_name << "]=";

	while (std::getline(ss, token, ','))
	{
		const auto val = std::stof(token);
		std::cout << val << " ";
		v.push_back(val);
	}
	std::cout << std::endl;

	return v;
}


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
		"^[#;]"						// comment lines we need to ignore
		"|"
		"^\\[\\s*([^\\]]+?)\\s*\\]"	// group #1:  new section, such as "[net]"
		"|"							// ...or...
		"^(\\S+)"					// group #2:  key
		"\\s*=\\s*"					// =
		"([^#]*)"					// group #3:  optional value
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
		const bool found = std::regex_search(line, m, rx);
		if (not found)
		{
			throw std::logic_error("failed to parse line #" + std::to_string(line_number) + " in " + cfg_filename.string());
		}

		const std::string section_name	= lowercase(m.str(1));
		const std::string key			= lowercase(m.str(2));
		const std::string val			= m.str(3);

		if (section_name.empty() and key.empty())
		{
			// ignore comments
			continue;
		}

		if (not section_name.empty())
		{
			// new section found
			sections.emplace_back(section_name, line_number);
			continue;
		}

		// we have a key-value pair...but make sure we at least have a section before we attempt to add a new key-value pair

		if (sections.empty())
		{
			throw std::logic_error("config cannot have values prior to \"[...]\" section name at line " + std::to_string(line_number));
		}

		// add this key-pair to the *most recent* section that we created
		Section & section = *sections.rbegin();
		if (section.kv_pairs.count(key) > 0)
		{
			throw std::logic_error("[" + section.name + "] already contains " + key + "=" + section.kv_pairs[key] + ", but duplicate key found on line #" + std::to_string(line_number));
		}
		section.kv_pairs[key] = val;
	}

	if (empty())
	{
		throw std::invalid_argument("configuration file is empty: \"" + cfg_filename.string() + "\"");
	}

	if (layer_type_from_string(sections[0].name) != ELayerType::kNetwork)
	{
		throw std::logic_error("configuration file must start with [net] or [network] section: " + cfg_filename.string() + "\"");
	}

	return *this;
}


size_t Darknet_ng::Config::count(const std::string & name) const
{
	const auto section_name = lowercase(name);

	return std::count_if(sections.begin(), sections.end(),
		 [&](const Section & s)
		 {
			 return s.name == section_name;
		 });
}


const Darknet_ng::Section & Darknet_ng::Config::find(const std::string & name) const
{
	const auto section_name = lowercase(name);

	for (const auto & s : sections)
	{
		if (s.name == section_name)
		{
			return s;
		}
	}

	throw std::invalid_argument("configuration does not have a section named \"" + section_name + "\"");
}


const Darknet_ng::Section & Darknet_ng::Config::find_next(const std::string & name, const Section & previous_section) const
{
	Sections::const_iterator iter;
	for (iter = sections.begin(); iter != sections.end(); ++ iter)
	{
		if (*iter == previous_section)
		{
			// we've now found the "previous section"
			break;
		}
	}

	if (iter == sections.end())
	{
		throw std::invalid_argument("failed to find the exact section \"" + previous_section.name + "\" in the configuration");
	}

	// if we get here, then we've found "previous_section" so all that remains is to find the next instance of the given name

	const auto section_name = lowercase(name);

	while (true)
	{
		iter ++;
		if (iter == sections.end())
		{
			break;
		}

		if (iter->name == section_name)
		{
			return *iter;
		}
	}

	throw std::invalid_argument("configuration does not have a section named \"" + section_name + "\" after the given section \"" + previous_section.name + "\"");
}


std::ostream & Darknet_ng::operator<<(std::ostream & os, const Darknet_ng::Section & section)
{
	os << "[" << section.name << "]";
	if (section.line_number > 0)
	{
		os << " # originally on line " << section.line_number;
	}

	os	<< std::endl
		<< "# keys: " << section.kv_pairs.size() << std::endl;

	for (const auto & [key, val] : section.kv_pairs)
	{
		std::cout << key << "=" << val << std::endl;
	}

	return os;
}


std::ostream & Darknet_ng::operator<<(std::ostream & os, const Darknet_ng::Config & cfg)
{
	std::cout << "# sections: " << cfg.sections.size() << std::endl;

	for (const auto & s : cfg.sections)
	{
		os << s << std::endl;
	}

	return os;
}
