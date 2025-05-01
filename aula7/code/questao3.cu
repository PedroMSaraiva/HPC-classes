#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initializeDecreasing(int *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = size - 1 - idx;
    }
}

__global__ void initializeIncreasing(int *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = idx;
    }
}

__global__ void addVectors(int *vecA, int *vecB, int *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = vecA[idx] + vecB[idx];
    }
}

bool checkAllEqual(int *vec, int size) {
    int firstValue = vec[0];
    for (int i = 1; i < size; i++) {
        if (vec[i] != firstValue) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1000;
    int *vecA, *vecB, *result;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&vecA, size);
    cudaMallocManaged(&vecB, size);
    cudaMallocManaged(&result, size);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    initializeDecreasing<<<blocksPerGrid, threadsPerBlock>>>(vecA, N);
    initializeIncreasing<<<blocksPerGrid, threadsPerBlock>>>(vecB, N);
    cudaDeviceSynchronize();

    
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(vecA, vecB, result, N);
    cudaDeviceSynchronize();

    if (checkAllEqual(result, N)) {
        printf("Sucesso! Todos os valores são iguais: %d\n", result[0]);
    } else {
        printf("Erro! Os valores não são todos iguais.\n");
    }

    cudaFree(vecA);
    cudaFree(vecB);
    cudaFree(result);

    return 0;
} 