# Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
# MIT license applies.  See "license.txt" for details.


FIND_PACKAGE (Threads			REQUIRED)
FIND_PACKAGE (OpenCV	CONFIG	REQUIRED)

INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
