// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#include "darknet-ng.hpp"


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void Darknet_ng::repack_input(float * input, float * re_packed_input, int w, int h, int c)
{
	const int items_per_channel = w * h;

	for (int chan = 0; chan < c; chan += 32)
	{
		for (int i = 0; i < items_per_channel; ++i)
		{
			for (int c_pack = 0; c_pack < 32; ++c_pack)
			{
				float src = input[(chan + c_pack) * items_per_channel + i];
				re_packed_input[chan * items_per_channel + i * 32 + c_pack] = src;
			}
		}
	}

	return;
}


void Darknet_ng::float_to_bit(float *src, unsigned char *dst, const size_t size)
{
	size_t dst_size = size / 8 + 1;
	memset(dst, 0, dst_size);

	//__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
	__m256 float_zero256 = _mm256_set1_ps(0.0);

	for (size_t i = 0; i < size; i += 8)
	{
		//__m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
		//__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
		//uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
		////mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

		__m256 src256 = _mm256_loadu_ps((float *)(&src[i]));
		__m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
		uint32_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

		dst[i / 8] = mask;
	}

	return;
}


void im2col_cpu_custom(float * data_im, int channels, int height, int width, int ksize, int stride, int pad, float * data_col)
{
	int c;
	const int height_col = (height + 2 * pad - ksize) / stride + 1;
	const int width_col = (width + 2 * pad - ksize) / stride + 1;
	const int channels_col = channels * ksize * ksize;

	// optimized version
	if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
	{
		#pragma omp parallel for
		for (c = 0; c < channels_col; ++c) {
			int h, w;
			int w_offset = c % ksize;
			int h_offset = (c / ksize) % ksize;
			int c_im = c / ksize / ksize;
			for (h = pad; h < height_col-pad; ++h)
			{
				for (w = pad; w < width_col-pad-8; w += 8)
				{
					int im_row = h_offset + h - pad;
					int im_col = w_offset + w - pad;
					int col_index = (c * height_col + h) * width_col + w;

					//data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
					__m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
					_mm256_storeu_ps(&data_col[col_index], src256);
				}

				for (; w < width_col - pad; ++w)
				{
					int im_row = h_offset + h - pad;
					int im_col = w_offset + w - pad;
					int col_index = (c * height_col + h) * width_col + w;

					data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
				}
			}

			{
				w = 0;
				for (h = 0; h < height_col; ++h)
				{
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					int col_index = (c * height_col + h) * width_col + w;
					data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
				}
			}

			{
				w = width_col-1;
				for (h = 0; h < height_col; ++h)
				{
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					int col_index = (c * height_col + h) * width_col + w;
					data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
				}
			}

			{
				h = 0;
				for (w = 0; w < width_col; ++w)
				{
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					int col_index = (c * height_col + h) * width_col + w;
					data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
				}
			}

			{
				h = height_col-1;
				for (w = 0; w < width_col; ++w)
				{
					int im_row = h_offset + h;
					int im_col = w_offset + w;
					int col_index = (c * height_col + h) * width_col + w;
					data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
				}
			}
		}

	}
	else
	{
		//printf("\n Error: is no non-optimized version \n");
		im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
	}

	return;
}


void Darknet_ng::transpose_uint32(uint32_t * src, uint32_t * dst, int src_h, int src_w, int src_align, int dst_align)
{
	//l.bit_align - algined (n) by 32
	//new_ldb - aligned (k) by 256

	//#pragma omp parallel for
	for (int i = 0; i < src_h; i += 1)  // l.size*l.size*l.c;
	{
		for (int j = 0; j < src_w; j += 1)  // out_h*out_w;
		{
			((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
		}
	}
}


void Darknet_ng::gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED, unsigned char * A, int lda, unsigned char * B, int ldb, float * C, int ldc, float * mean_arr)
{
	#if defined(_OPENMP)
	static int max_num_threads = 0;
	if (max_num_threads == 0) {
		max_num_threads = omp_get_max_threads();
		//omp_set_num_threads(max_num_threads / 2);
	}
	#endif

	//#pragma omp parallel for
	//for (i = 0; i < M; ++i)
	#pragma omp parallel for
	for (int i = 0; i < (M/2)*2; i += 2)
	{   // l.n - filters [16 - 55 - 1024]
		float mean_val_0 = mean_arr[i + 0];
		float mean_val_1 = mean_arr[i + 1];
		//__m256i all_1 = _mm256_set1_epi8(255);

		//for (j = 0; j < N; ++j)
		for (int j = 0; j < (N/2)*2; j += 2)
		{ // out_h*out_w - one channel output size [169 - 173056]
			//int count = 0;
			const int bit_step = 256;
			__m256i count_sum_0 = _mm256_set1_epi8(0);
			__m256i count_sum_1 = _mm256_set1_epi8(0);
			__m256i count_sum_2 = _mm256_set1_epi8(0);
			__m256i count_sum_3 = _mm256_set1_epi8(0);

			for (int k = 0; k < K; k += bit_step)
			{   // l.size*l.size*l.c - one filter size [27 - 9216]
				__m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + 0)*lda + k) / 8));
				__m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));

				__m256i a_bit256_1 = _mm256_loadu_si256((__m256i *)(A + ((i + 1)*lda + k) / 8));
				__m256i b_bit256_1 = _mm256_loadu_si256((__m256i *)(B + ((j + 1)*ldb + k) / 8));


				xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum_0);
				xnor_avx2_popcnt(a_bit256_0, b_bit256_1, &count_sum_1);

				xnor_avx2_popcnt(a_bit256_1, b_bit256_0, &count_sum_2);
				xnor_avx2_popcnt(a_bit256_1, b_bit256_1, &count_sum_3);

				//count += popcnt256(c_bit256);
				//binary_int64_printf(c_bit64);
				//printf(", count = %d \n\n", tmp_count);
			}

			int count_0 = get_count_mula(count_sum_0);
			int count_1 = get_count_mula(count_sum_1);
			int count_2 = get_count_mula(count_sum_2);
			int count_3 = get_count_mula(count_sum_3);

			const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
			count_0 = count_0 - f1;    // remove extra bits (from empty space for align only)
			count_1 = count_1 - f1;
			count_2 = count_2 - f1;
			count_3 = count_3 - f1;
			C[i*ldc + (j + 0)] = (2 * count_0 - K) * mean_val_0;
			C[i*ldc + (j + 1)] = (2 * count_1 - K) * mean_val_0;
			C[(i + 1)*ldc + (j + 0)] = (2 * count_2 - K) * mean_val_1;
			C[(i + 1)*ldc + (j + 1)] = (2 * count_3 - K) * mean_val_1;
		}

		for (int i_d = 0; i_d < 2; ++i_d)
		{
			float mean_val = mean_arr[i + i_d];
			for (j = (N / 2) * 2; j < N; j += 1)
			{ // out_h*out_w - one channel output size [169 - 173056]
				const int bit_step = 256;
				__m256i count_sum = _mm256_set1_epi8(0);

				for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
					__m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + i_d + 0)*lda + k) / 8));
					__m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));
					xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
				}
				int count = get_count_mula(count_sum);
				const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
				count = count - f1;    // remove extra bits (from empty space for align only)
				C[(i + i_d)*ldc + j] = (2 * count - K) * mean_val;
			}
		}
	}

	for (int i = (M / 2) * 2; i < M; i += 1)
	{
		float mean_val = mean_arr[i];
		int j, k;
		for (j = 0; j < N; j += 1)
		{ // out_h*out_w - one channel output size [169 - 173056]
			const int bit_step = 256;
			__m256i count_sum = _mm256_set1_epi8(0);

			for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
				__m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + 0)*lda + k) / 8));
				__m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));
				xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
			}
			int count = get_count_mula(count_sum);
			const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
			count = count - f1;    // remove extra bits (from empty space for align only)
			C[i*ldc + j] = (2 * count - K) * mean_val;
		}
	}

	return;
}
