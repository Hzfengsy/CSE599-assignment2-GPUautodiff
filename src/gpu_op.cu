#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
using std::swap;

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void arrayset_kernel(int n, float value, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	output[idx] = value;
}

__global__ void broadcast_to_kernel(int n, int m, const float *input, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= m) return;
	for (int i = 0; i < n; i++) output[idx + i * m] = input[idx];
}

__global__ void elementwise_add_kernel(int n, const float *input_a, const float *input_b, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	output[idx] = input_a[idx] + input_b[idx];
}

__global__ void elementwise_add_const_kernel(int n, const float *input, float val, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	output[idx] = input[idx] + val;
}

__global__ void reduce_sum_axis_zero_kernel(int n, int nrow, int ncol, const float *input, float *output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= ncol || y >= nrow) return;
	int idx = y * ncol + x, step = nrow * ncol;
	float sum = 0;
	for (int i = 0; i < n; i++) sum += input[idx + i * step];
	output[idx] = sum;
}

__global__ void elementwise_multi_kernel(int n, const float *input_a, const float *input_b, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	output[idx] = input_a[idx] * input_b[idx];
}

__global__ void elementwise_multi_const_kernel(int n, const float *input, float val, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	output[idx] = input[idx] * val;
}

__global__ void relu_kernel(int n, const float *input, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	if (input[idx] > 0) output[idx] = input[idx];
	else output[idx] = 0;
}

__global__ void relu_grad_kernel(int n, const float *input, const float *grad, float *output)
{
	int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	if (input[idx] > 0) output[idx] = grad[idx];
	else if (input[idx] == 0) output[idx] = 0.5 * grad[idx];
	else output[idx] = 0;
}

__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output)
{
	int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (y >= nrow) return;
	input += y * ncol;
	output += y * ncol;
	float maxval = *input, sum = 0;
	for (int x = 1; x < ncol; ++x) maxval = max(maxval, input[x]);
	for (int x = 0; x < ncol; ++x) sum += exp(input[x] - maxval);
	for (int x = 0; x < ncol; ++x) output[x] = exp(input[x] - maxval) / sum;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void softmax_cross_entropy_kernel(int nrow, int ncol,
											const float *input_a,
											const float *input_b,
											float *output)
{
	// Dynamic shared memory, size provided at kernel launch.
	extern __shared__ float loss_per_row[];
	// Two dimensional thread blocks.
	int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (y >= nrow) return;
	input_a += y * ncol;
	input_b += y * ncol;
	float maxval = *input_a;
	// Find max for a row.
	for (int x = 1; x < ncol; ++x) maxval = max(maxval, input_a[x]);
	// Deduct by max for a row, and raise to exp.
	float sum = 0;
	for (int x = 0; x < ncol; ++x) sum += exp(input_a[x] - maxval);
	// Compute per-row loss.
	float loss = 0;
	for (int x = 0; x < ncol; ++x) loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
	loss_per_row[y] = loss;     
	__syncthreads();
	// Compute reduce_mean across rows.
	float mean_loss = 0;
	// Use a single thread to reduce mean across rows.
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		for (int i = 0; i < nrow; ++i) mean_loss += loss_per_row[i];
		mean_loss /= nrow;
		output[0] = mean_loss;
	}
}

int DLGpuArraySet(DLArrayHandle arr, float value)
{ 
	float *output_data = (float *)arr->data;
	int n = 1;
	for (int i = 0; i < arr->ndim; i++) n *= arr->shape[i];
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	arrayset_kernel<<<blocks, threads>>>(n, value, output_data);
	return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	int n = output->shape[0];
	int m = 1;
	for (int i = 0; i < input->ndim; i++) m *= input->shape[i];
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((m + size - 1) / size);
	broadcast_to_kernel<<<blocks, threads>>>(n, m, input_data, output_data);
	return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	int n = input->shape[0];
	int nrow = input->shape[1];
	int ncol = input->shape[2];
	int size = 32;
	dim3 threads(size, size);
	dim3 blocks((ncol + size - 1) / size, (nrow + size - 1) / size);
	reduce_sum_axis_zero_kernel<<<blocks, threads>>>(n, nrow, ncol, input_data, output_data);
	return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output)
{
	const float *input_dataA = (const float *)matA->data;
	const float *input_dataB = (const float *)matB->data;
	float *output_data = (float *)output->data;
	int n = 1;
	assert(matA->ndim == matB->ndim);
	for (int i = 0; i < matA->ndim; i++) 
		if (matA->shape[i] == matB->shape[i]) n *= matA->shape[i];
		else n = 0;
	assert(n > 0);
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	elementwise_add_kernel<<<blocks, threads>>>(n, input_dataA, input_dataB, output_data);
	return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val, DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	int n = 1;
	for (int i = 0; i < input->ndim; i++) n *= input->shape[i];
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	elementwise_add_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
	return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output)
{
	const float *input_dataA = (const float *)matA->data;
	const float *input_dataB = (const float *)matB->data;
	float *output_data = (float *)output->data;
	int n = 1;
	assert(matA->ndim == matB->ndim);
	for (int i = 0; i < matA->ndim; i++) 
		if (matA->shape[i] == matB->shape[i]) n *= matA->shape[i];
		else n = 0;
	assert(n > 0);
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	elementwise_multi_kernel<<<blocks, threads>>>(n, input_dataA, input_dataB, output_data);
	return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val, DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	int n = 1;
	for (int i = 0; i < input->ndim; i++) n *= input->shape[i];
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	elementwise_multi_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
	return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
						const DLArrayHandle matB, bool transposeB,
						DLArrayHandle matC)
{
	// 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
	assert (status == CUBLAS_STATUS_SUCCESS);
	cudaThreadSynchronize();
	int nrow_A = matA->shape[0];
	int ncol_A = matA->shape[1];
	int nrow_B = matB->shape[0];
	int ncol_B = matB->shape[1];
	if (transposeA) swap(nrow_A, ncol_A);
	if (transposeB) swap(nrow_B, ncol_B);

	cublasOperation_t trans_A = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t trans_B = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
	const float *input_data_A = (const float *)matA->data;
	const float *input_data_B = (const float *)matB->data;
	float *output_data = (float *)matC->data;
	assert(nrow_A == matC->shape[0] && ncol_B == matC->shape[1]);
	assert(ncol_A == nrow_B);

    float a = 1, b = 0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm (
        handle,    			// blas 库对象 
        trans_B,			// 矩阵 A 属性参数
        trans_A,			// 矩阵 B 属性参数
        ncol_B,				// A, C 的行数 
        nrow_A,				// B, C 的列数
        nrow_B,				// A 的列数和 B 的行数
        &a,					// 运算式的 α 值
        input_data_B,		// A 在显存中的地址
        transposeB ? nrow_B : ncol_B,				// lda
        input_data_A,		// B 在显存中的地址
        transposeA ? nrow_A : ncol_A,				// ldb
        &b,					// 运算式的 β 值
     	output_data,				// C 在显存中的地址(结果矩阵)
        ncol_B				// ldc
    );
	
	cudaThreadSynchronize();
	return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	int n = 1;
	for (int i = 0; i < input->ndim; i++) n *= input->shape[i];
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	relu_kernel<<<blocks, threads>>>(n, input_data, output_data);
	return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
					  DLArrayHandle output)
{
	const float *input_data = (const float *)input->data;
	const float *grad_data = (const float *)in_grad->data;
	float *output_data = (float *)output->data;
	int n = 1;
	assert(input->ndim == in_grad->ndim);
	for (int i = 0; i < input->ndim; i++) 
		if (input->shape[i] == in_grad->shape[i]) n *= input->shape[i];
		else n = 0;
	assert(n > 0);
	int size = 1024;
	dim3 threads(size);
	dim3 blocks((n + size - 1) / size);
	relu_grad_kernel<<<blocks, threads>>>(n, input_data, grad_data, output_data);
	return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output)
{
	assert(input->ndim == 2);
	int nrow = input->shape[0];
	assert(nrow <= 1024 * 4);
	int ncol = input->shape[1];
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	dim3 threads;
	if (nrow <= 1024) threads.x = nrow;
	else threads.x = 1024, threads.y = (nrow + 1023) / 1024;
	softmax_kernel<<<1, threads>>>(nrow, ncol, input_data, output_data);
	return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
							 const DLArrayHandle input_b,
							 DLArrayHandle output)
{
	assert(input_a->ndim == 2);
	assert(input_b->ndim == 2);
	// assert(output->ndim == 1);
	assert(input_a->shape[0] == input_b->shape[0] && input_a->shape[1] == input_b->shape[1]);
	int nrow = input_a->shape[0];
	// Maximum x- or y-dimension of a block = 1024
	// But we need 'nrow' shared memory, and max shared memory is 48KB.
	// Conservatively allow max 16KB shared memory.
	assert(nrow <= 1024 * 4);
	int ncol = input_a->shape[1];
	const float *input_data_a = (const float *)input_a->data;
	const float *input_data_b = (const float *)input_b->data;
	float *output_data = (float *)output->data;
	dim3 threads;
	if (nrow <= 1024) threads.x = nrow;
	else threads.x = 1024, threads.y = (nrow + 1023) / 1024;
	// 1 block, each block with 'threads' number of threads with 'nrow' shared
	// memory size
	softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>
		(nrow, ncol, input_data_a, input_data_b, output_data);
	return 0;
}


