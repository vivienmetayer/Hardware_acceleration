#pragma once
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements);
__global__ void detect_duplicates(const float* in, bool* duplicates, int numElements);
__global__ void detect_duplicates_v2(const float* in, bool* duplicates, int numElements);
__global__ void detect_duplicates_v3(const float* in, bool* duplicates, int numElements);
int cuda_test(void);
std::vector<bool> detect_duplicates_cuda(std::vector<int>& v);
std::vector<bool> detect_duplicates_cuda_v2(std::vector<int>& v);
std::vector<bool> detect_duplicates_cuda_v3(std::vector<int>& v);