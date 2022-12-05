// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


namespace Darknet_ng
{
	void repack_input(float * input, float * re_packed_input, int w, int h, int c);
	void float_to_bit(float * src, unsigned char * dst, const size_t size);

	/// From Berkeley Vision's Caffe!  https://github.com/BVLC/caffe/blob/master/LICENSE
	void im2col_cpu_custom(float * data_im, int channels, int height, int width, int ksize, int stride, int pad, float * data_col);

	void transpose_uint32(uint32_t * src, uint32_t * dst, int src_h, int src_w, int src_align, int dst_align);

	/** 5x times faster than gemm()-float32.
	 * @todo Further optimizations: do mean-mult only for the last layer
	 */
	void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED, unsigned char * A, int lda, unsigned char * B, int ldb, float * C, int ldc, float * mean_arr);

	/// Two versions of this function exists -- CPU and GPU.
	void im2col_cpu_custom_bin(float * data_im, int channels, int height, int width, int ksize, int stride, int pad, float * data_col, int bit_align);
}
