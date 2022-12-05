// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"

#ifdef GPU /// @todo where to set GPU so we only include one of CPU or GPU?


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void Darknet_ng::im2col_cpu_custom_bin(float * data_im, int channels, int height, int width, int ksize, int stride, int pad, float * data_col, int bit_align)
{
	int c;
	const int height_col = (height + 2 * pad - ksize) / stride + 1;
	const int width_col = (width + 2 * pad - ksize) / stride + 1;
	const int channels_col = channels * ksize * ksize;

	// optimized version
	if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
	{
		__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
		__m256 float_zero256 = _mm256_set1_ps(0.00);

		int new_ldb = bit_align;

		#pragma omp parallel for
		for (c = 0; c < channels_col; ++c) {
			int h, w;
			int w_offset = c % ksize;
			int h_offset = (c / ksize) % ksize;
			int c_im = c / ksize / ksize;
			for (h = pad; h < height_col - pad; ++h) {
				for (w = pad; w < width_col - pad - 8; w += 8) {
					int im_row = h_offset + h - pad;
					int im_col = w_offset + w - pad;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//__m256i src256 = _mm256_loadu_si256((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
					//__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
					//uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
					//mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

					__m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
					__m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
					uint16_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

					uint16_t* dst_ptr = (uint16_t*)&((uint8_t*)data_col)[col_index / 8];
					*dst_ptr |= (mask << (col_index % 8));
				}

				for (; w < width_col - pad; ++w) {
					int im_row = h_offset + h - pad;
					int im_col = w_offset + w - pad;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
					float val = data_im[im_col + width*(im_row + height*c_im)];
					if (val > 0) set_bit((unsigned char* const)data_col, col_index);
				}
			}

			{
				w = 0;
				for (h = 0; h < height_col; ++h) {
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					if (val > 0) set_bit((unsigned char* const)data_col, col_index);
				}
			}

			{
				w = width_col - 1;
				for (h = 0; h < height_col; ++h) {
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					if (val > 0) set_bit((unsigned char* const)data_col, col_index);
				}
			}

			{
				h = 0;
				for (w = 0; w < width_col; ++w) {
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					if (val > 0) set_bit((unsigned char* const)data_col, col_index);
				}
			}

			{
				h = height_col - 1;
				for (w = 0; w < width_col; ++w) {
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					//int col_index = (c * height_col + h) * width_col + w;
					int col_index = c * new_ldb + h * width_col + w;

					//data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
					if (val > 0) set_bit((unsigned char* const)data_col, col_index);
				}
			}
		}

	}
	else {
		printf("\n Error: is no non-optimized version \n");
		//im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
		// float_to_bit(b, t_input, src_size);
		// transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
	}
}

#endif
