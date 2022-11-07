# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

PROJECT (DarknetNG C CXX)

IF (NOT CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Release)
ENDIF ()

INCLUDE (CM_version.cmake		) # get the version number
INCLUDE (CM_definitions.cmake	) # compiler definitions
INCLUDE (CM_dependencies.cmake	) # find all required dependencies
INCLUDE (CM_source.cmake		) # source code that needs to be built
INCLUDE (CM_package.cmake		) # create .exe or .deb packages

