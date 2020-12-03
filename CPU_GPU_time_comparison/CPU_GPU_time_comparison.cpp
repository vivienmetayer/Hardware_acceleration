#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cmath>
#include <mutex>
#include <xlocmon>
#include <cuda_runtime.h>

#include "kernels.h"

std::mutex m;
int num_threads = 24;

std::vector<bool> detect_duplicates_CPU_simple(std::vector<int>& v)
{
	int size = v.size();
	std::vector<bool> duplicates(size, false);
	
	for (int i = 0; i < size-1; ++i) {
		for (int j = i+1; j < size; ++j) {
			if (v[i] == v[j]) {
				duplicates[i] = true;
				duplicates[j] = true;
				break;
			}
		}
	}
	
	return duplicates;
}

std::vector<bool> detect_duplicates_CPU_std(std::vector<int>& v)
{
	int size = v.size();
	std::vector<bool> duplicates(size, false);
	
	for (int i = 0; i < size - 1; ++i) {
		auto it = std::find(v.begin() + i + 1, v.end(), v[i]);
		if (it != v.end()) {
			duplicates[i] = true;
			duplicates[std::distance(v.begin(), it)] = true;
		}
	}
	
	return duplicates;
}

void detect_duplicates_CPU_thread(std::vector<int>& v, int start_index, int end_index, int max_size, std::vector<bool>& duplicates)
{
	for (int i = start_index; i < end_index; ++i) {
		if (i >= max_size) return;
		for (int j = i + 1; j < max_size; ++j) {
			if (j >= max_size) continue;
			if (v[i] == v[j]) {
				duplicates[i] = true;
				duplicates[j] = true;
				break;
			}
		}
	}
}

void detect_duplicates_CPU_multithread(std::vector<int>& v, std::vector<bool>& duplicates)
{
	long long size = v.size();

	std::vector<std::thread> threads;
	for (int i = 0; i < num_threads; ++i) {
		const int thread_size = static_cast<int>(ceil(size / num_threads));
		int start = i * thread_size;
		int end = start + thread_size;
		threads.emplace_back(detect_duplicates_CPU_thread, std::ref(v),start, end, size, std::ref(duplicates));
	}
	for (auto &t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
}

void detect_duplicates_CPU_thread_std(std::vector<int>& v, int start_index, int end_index, int max_size, std::vector<bool>& duplicates)
{
	for (int i = start_index; i < end_index; ++i) {
		if (i >= max_size) return;
		auto it = std::find(v.begin() + i + 1, v.end(), v[i]);
		if (it != v.end()) {
			duplicates[i] = true;
			duplicates[std::distance(v.begin(), it)] = true;
		}
	}
}

void detect_duplicates_CPU_multithread_std(std::vector<int>& v, std::vector<bool>& duplicates)
{
	long long size = v.size();

	std::vector<std::thread> threads;
	for (int i = 0; i < num_threads; ++i) {
		const int thread_size = static_cast<int>(ceil(size / num_threads));
		int start = i * thread_size;
		int end = start + thread_size;
		threads.emplace_back(detect_duplicates_CPU_thread_std, std::ref(v), start, end, size, std::ref(duplicates));
	}
	for (auto& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
}

void print(__m128 sse)
{
	std::cout << sse.m128_f32[0] << ' ';
	std::cout << sse.m128_f32[1] << ' ';
	std::cout << sse.m128_f32[2] << ' ';
	std::cout << sse.m128_f32[3] << std::endl;
}

void print(__m256i sse)
{
	std::cout << sse.m256i_i32[0] << ' ';
	std::cout << sse.m256i_i32[1] << ' ';
	std::cout << sse.m256i_i32[2] << ' ';
	std::cout << sse.m256i_i32[3] << ' ';
	std::cout << sse.m256i_i32[4] << ' ';
	std::cout << sse.m256i_i32[5] << ' ';
	std::cout << sse.m256i_i32[6] << ' ';
	std::cout << sse.m256i_i32[7] << std::endl;
}

std::vector<bool> detect_duplicates_CPU_SIMD(std::vector<int> &v)
{	
	long long size = v.size();
	std::vector<int> v2(size + 8, 0);
	for (int i = 0; i < 8; ++i) {
		v.push_back(-1);
	}
	std::vector<bool> duplicates(size, false);

	for (int i = 0; i < size-1; ++i) {
		long long rest = size - i - 1;
		int num_m256 = ceil(rest / 8.0);
		
		// Set element to compare in vector
		__m256i element = _mm256_set1_epi32(v[i]);
		
		for (int j = 0; j < num_m256; ++j) {
			int index = i + 1 + j * 8;

			// Load next 8 values
			__m256i data = _mm256_loadu_epi32(&v[index]);

			// Compare
			__m256i t0 = _mm256_cmpeq_epi32(data, element);

			// If at least one duplicate found
			if (_mm256_movemask_epi8(t0) != 0) {
				v2[i] = -1;
				// load data from v2, do a OR to add the result
				__m256i data_v2 = _mm256_loadu_epi32(&v2[index]);
				data_v2 = _mm256_or_si256(data_v2, t0);
				// Store the result
				_mm256_storeu_epi32(&v2[index], data_v2);
				break;
			}
		}
	}
	// Transform v2 into a bool vector
	std::transform(v2.begin(), v2.end() - 8, duplicates.begin(), [](const int a) {return a != 0; });
	return duplicates;
}

void detect_duplicates_CPU_SIMD_thread(std::vector<int>& v, std::vector<int>& v2, int start_index, int end_index, int max_size, std::vector<bool>& duplicates)
{
	for (int i = start_index; i < end_index; ++i) {
		if (i >= max_size) return;
		int rest = max_size - i - 1;
		int num_m256 = ceil(rest / 8.0);

		// Set element to compare in vector
		__m256i element = _mm256_set1_epi32(v[i]);
		
		for (int j = 0; j < num_m256; ++j) {
			int index = i + 1 + j * 8;

			// Load next 8 values
			__m256i data = _mm256_loadu_epi32(&v[index]);

			// Compare
			__m256i t0 = _mm256_cmpeq_epi32(data, element);

			// If at least one duplicate found
			if (_mm256_movemask_epi8(t0) != 0) {
				v2[i] = 1;
				// load data from v2, do a OR to add the result
				__m256i data_v2 = _mm256_loadu_epi32(&v2[index]);
				data_v2 = _mm256_or_si256(data_v2, t0);
				// Store the result
				_mm256_storeu_epi32(&v2[index], data_v2);
				break;
			}
		}
	}
}

std::vector<bool> detect_duplicates_CPU_SIMD_multithread(std::vector<int>& v)
{
	// Add 8 elements to prevent reading outside vector
	long long size = v.size();
	std::vector<int> v2(size + 8, 0);
	for (int i = 0; i < 8; ++i) {
		v.push_back(-1);
	}
	std::vector<bool> duplicates(size, false);

	// launch threads
	std::vector<std::thread> threads;
	for (int i = 0; i < num_threads; ++i) {
		const int thread_size = static_cast<int>(ceil(size / num_threads));
		int start = i * thread_size;
		int end = start + thread_size;
		threads.emplace_back(detect_duplicates_CPU_SIMD_thread, std::ref(v), std::ref(v2), start, end, size, std::ref(duplicates));
	}

	// join threads
	for (auto& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}

	// transform to get a bool array, ignoring the last 8 elements added at the beginning
	std::transform(v2.begin(), v2.end() - 8, duplicates.begin(), [](const int a) {return a != 0; });
	return duplicates;
}

std::vector<bool> detect_duplicates_GPU_CUDA(std::vector<int>& v)
{
	return detect_duplicates_cuda(v);
}

std::vector<bool> detect_duplicates_GPU_CUDA_v2(std::vector<int>& v)
{
	return detect_duplicates_cuda_v2(v);
}

std::vector<bool> detect_duplicates_GPU_CUDA_v3(std::vector<int>& v)
{
	return detect_duplicates_cuda_v3(v);
}

void detect_duplicates_GPU_SYCL()
{

}

int main()
{
	const int vec_size = 200000;
	std::cout << "number of elements : " << vec_size << std::endl;

	// init random
	const unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
	std::mt19937 gen(seed);
	std::uniform_int_distribution<unsigned int> dist(0, vec_size * 10);

	// init vector of int
	std::vector<int> v(vec_size);
	for (int i = 0; i < vec_size; ++i) {
		v[i] = (dist(gen));
	}

	auto t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates = detect_duplicates_CPU_simple(v);
	auto t1 = std::chrono::system_clock::now();
	std::cout << "base CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates_std = detect_duplicates_CPU_std(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "Std CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
	
	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates2(vec_size, false);
	detect_duplicates_CPU_multithread(v, duplicates2);
	t1 = std::chrono::system_clock::now();
	std::cout << "Multi thread CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates3(vec_size, false);
	detect_duplicates_CPU_multithread_std(v, duplicates3);
	t1 = std::chrono::system_clock::now();
	std::cout << "Multi thread std CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates4 = detect_duplicates_CPU_SIMD(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "SIMD CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
	v.resize(vec_size);
	
	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates5 = detect_duplicates_CPU_SIMD_multithread(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "SIMD Multithread CPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
	v.resize(vec_size);
	
	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates6 = detect_duplicates_GPU_CUDA(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "CUDA v1 GPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

	/*t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates7 = detect_duplicates_GPU_CUDA_v2(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "CUDA v2 GPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;

	t0 = std::chrono::system_clock::now();
	std::vector<bool> duplicates8 = detect_duplicates_GPU_CUDA_v3(v);
	t1 = std::chrono::system_clock::now();
	std::cout << "CUDA v3 GPU Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;*/
	
	for (int i = 0; i < vec_size; ++i) {
		if (duplicates[i] != duplicates2[i]) {
			std::cout << "error 1" << std::endl;
		}
		if (duplicates[i] != duplicates_std[i]) {
			std::cout << "error 2" << std::endl;
		}
		if (duplicates[i] != duplicates3[i]) {
			std::cout << "error 3" << std::endl;
		}
		if (duplicates[i] != duplicates4[i]) {
			std::cout << "error 4, at index " << i << ", should be " << duplicates[i] << " but is " << duplicates4[i] << " with value " << v[i] << std::endl;
		}
		/*if (duplicates[i] != duplicates5[i]) {
			std::cout << "error 5, at index " << i << ", should be " << duplicates[i] << " but is " << duplicates5[i] << " with value " << v[i] << std::endl;
		}
		if (duplicates[i] != duplicates6[i]) {
			std::cout << "error 6, at index " << i << ", should be " << duplicates[i] << " but is " << duplicates6[i] << " with value " << v[i] << std::endl;
		}*/
	}
	
	return 0;
}