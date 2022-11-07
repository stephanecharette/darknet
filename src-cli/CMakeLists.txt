# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


ADD_EXECUTABLE			(darknet-ng-cli main.cpp darknet-ng.rc)
TARGET_LINK_LIBRARIES	(darknet-ng-cli PRIVATE Threads::Threads darknet-ng ${OpenCV_LIBS})
INSTALL					(TARGETS darknet-ng-cli DESTINATION bin)

IF ( WIN32 )
	INSTALL (FILES darknet-ng.ico DESTINATION bin)
ENDIF ()

