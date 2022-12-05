// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


void Darknet_ng::activate_array_cpu_custom(float * x, const int n, const Darknet_ng::EActivation a)
{
	int i = 0;
	if (a == EActivation::kLinear)
	{
		// do nothing
	}
	else if (a == EActivation::kLeaky)
	{
		if (is_fma_avx2())
		{
			__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
			__m256 all256_01 = _mm256_set1_ps(0.1F);

			for (i = 0; i < n - 8; i += 8)
			{
				//x[i] = (x[i]>0) ? x[i] : .1*x[i];

				__m256 src256 = _mm256_loadu_ps(&x[i]);
				__m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

				__m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

				__m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
				_mm256_storeu_ps(&x[i], result256);
			}
		}

		for (; i < n; ++i)
		{
			x[i] = (x[i] > 0.0f) ? x[i] : 0.1f * x[i];
		}
	}
	else
	{
		for (i = 0; i < n; ++i)
		{
			x[i] = activate(x[i], a);
		}
	}

	return;
}
