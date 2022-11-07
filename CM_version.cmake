# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


FILE (READ version.txt VERSION_TXT)
STRING (STRIP "${VERSION_TXT}" VERSION_TXT)
STRING (REGEX MATCHALL "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)$" OUTPUT ${VERSION_TXT})

SET (DNG_VER_MAJOR	${CMAKE_MATCH_1})
SET (DNG_VER_MINOR	${CMAKE_MATCH_2})
SET (DNG_VER_PATCH	${CMAKE_MATCH_3})
SET (DNG_VER_COMMIT	${CMAKE_MATCH_4})

SET (DNG_VERSION ${DNG_VER_MAJOR}.${DNG_VER_MINOR}.${DNG_VER_PATCH}-${DNG_VER_COMMIT})
MESSAGE ( "Building ver: ${DH_VERSION}" )

ADD_DEFINITIONS (-DNG_VERSION="${DNG_VERSION}")

