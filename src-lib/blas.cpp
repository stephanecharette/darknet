// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "blas.hpp"


void Darknet_ng::fill_cpu(const int n, float const alpha, float * ptr, const int incx)
{
	if (incx == 1 && alpha == 0.0f)
	{
		memset(ptr, 0, n * sizeof(float));
	}
	else
	{
		for (int i = 0; i < n; ++i)
		{
			ptr[i * incx] = alpha;
		}
	}

	return;
}

