// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


/** Macro to pass in the filename, function name, and line number to the Darknet NG exception class.
 * @todo Replace this with @p std::source_location once we can use C++20.
 */
#define DNG_LOC __FILE__, __func__, __LINE__


namespace Darknet_ng
{
	/** An exception that derives from the C++ @p std::runtime_error class, but also contains information on where the
	 * exception was created.  All functions and methods in the @ref Darknet_ng namespace should be using this exception
	 * class.
	 *
	 * @since 2022-12-03
	 */
	class Exception final : public std::runtime_error
	{
		public:

			/** Constructor.
			 * @param [in] msg This text is returned by @p std::exception::what().
			 * @param [in] fn This is the function where the exeption was created.  Use @ref DNG_LOC.
			 * @param [in] func This is the function where the exception was created.  Use @ref DNG_LOC.
			 * @param [in] line This is the line number where the exception was created.  Use @ref DNG_LOC.
			 *
			 * For example:
			 * ~~~~
			 * if (something)
			 * {
			 *     throw Exception("some message", DNG_LOC);
			 * }
			 * ~~~~
			 */
			Exception(const std::string & msg, const std::string & fn, const std::string & func, const int line);

			/// Destructor.
			virtual ~Exception();

			/// Return both the message and the location details.
			virtual const char * what() const noexcept;

			/// @{ Member usually filled in by the @ref DNG_LOC macro.
			std::string filename;
			std::string function;
			int line_number;
			/// @}
	};
}
