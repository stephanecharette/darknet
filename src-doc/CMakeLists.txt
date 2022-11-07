# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


FIND_PACKAGE (Doxygen QUIET)

IF (Doxygen_FOUND)

	CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile )

	ADD_CUSTOM_TARGET (
		doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating documentation with Doxygen" VERBATIM )

ENDIF ()

