# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


SET (CMAKE_POSITION_INDEPENDENT_CODE ON)
SET (BUILD_SHARED_LIBS TRUE)

FILE (GLOB SRC_LIB *.cpp	)
LIST (SORT SRC_LIB			)

FILE (GLOB HEADERS *.hpp	)
LIST (SORT HEADERS			)

ADD_LIBRARY (darknet-ng ${SRC_LIB})

INSTALL (FILES ${HEADERS} DESTINATION include)
INSTALL (TARGETS darknet-ng DESTINATION lib)
