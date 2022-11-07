# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


SET (CPACK_PACKAGE_VENDOR				"Stephane Charette"			)
SET (CPACK_PACKAGE_CONTACT				"stephanecharette@gmail.com")
SET (CPACK_PACKAGE_VERSION				${DNG_VERSION}				)
SET (CPACK_PACKAGE_VERSION_MAJOR		${DNG_VER_MAJOR}			)
SET (CPACK_PACKAGE_VERSION_MINOR		${DNG_VER_MINOR}			)
SET (CPACK_PACKAGE_VERSION_PATCH		${DNG_VER_PATCH}			)
SET (CPACK_RESOURCE_FILE_LICENSE		${CMAKE_CURRENT_SOURCE_DIR}/license.txt)
SET (CPACK_PACKAGE_NAME					"darknetng"					)
SET (CPACK_PACKAGE_DESCRIPTION_SUMMARY	"Darknet Next Gen"			)
SET (CPACK_PACKAGE_DESCRIPTION			"Darknet Next Gen - Darknet YOLO framework for computer vision / object detection")
SET (CPACK_PACKAGE_HOMEPAGE_URL			"https://github.com/stephanecharette/darknet/")
SET (CPACK_GENERATOR					"DEB"						)
SET (CPACK_SOURCE_IGNORE_FILES			".git" ".kdev4" "build/"	)
SET (CPACK_SOURCE_GENERATOR				"TGZ;ZIP"					)

INCLUDE (CPack)
