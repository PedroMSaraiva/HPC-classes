#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initializeConstant(int *vec, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = value;
    }
}

__global__ void initializeIncremental(int *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = idx;
    }
}

__global__ void multiplyVectors(int *vecA, int *vecB, int *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = vecA[idx] * vecB[idx];
    }
}

int main() {
    const int N = 70;
    int *vecA, *vecB, *result;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&vecA, size);
    cudaMallocManaged(&vecB, size);
    cudaMallocManaged(&result, size);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    initializeConstant<<<blocksPerGrid, threadsPerBlock>>>(vecA, N, 10);
    initializeIncremental<<<blocksPerGrid, threadsPerBlock>>>(vecB, N);
    cudaDeviceSynchronize();

    multiplyVectors<<<blocksPerGrid, threadsPerBlock>>>(vecA, vecB, result, N);
    cudaDeviceSynchronize();

    printf("Multiplicação dos vetores (elemento a elemento):\n");
    for (int i = 0; i < N; i++) {
        printf("%d * %d = %d\n", vecA[i], vecB[i], result[i]);
    }

    cudaFree(vecA);
    cudaFree(vecB);
    cudaFree(result);

    return 0;
} 