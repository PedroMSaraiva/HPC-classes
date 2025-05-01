#include <stdio.h>
#include <cuda_runtime.h>


__global__ void initializeVectors(int *vecA, int *vecB, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vecA[idx] = idx * 2;      
        vecB[idx] = idx * 3;      
    }
}

// Kernel para calcular V = k1*A + k2*B
__global__ void calculateV(int *vecA, int *vecB, int *vecV, int size, int k1, int k2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vecV[idx] = k1 * vecA[idx] + k2 * vecB[idx];
    }
}

int main() {
    const int N = 500;
    const int k1 = 2; 
    const int k2 = 3;  
    int *vecA, *vecB, *vecV;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&vecA, size);
    cudaMallocManaged(&vecB, size);
    cudaMallocManaged(&vecV, size);

    int threadsPerBlock = 16;
    int blocksPerGrid = 8;

    initializeVectors<<<blocksPerGrid, threadsPerBlock>>>(vecA, vecB, N);
    cudaDeviceSynchronize();

    calculateV<<<blocksPerGrid, threadsPerBlock>>>(vecA, vecB, vecV, N, k1, k2);
    cudaDeviceSynchronize();

    printf("Vetor V:\n");
    for (int i = 0; i < N; i++) {
        printf("V[%d] = %d\n", i, vecV[i]);
    }
    printf("Alguns valores serão zero, pois os numeros de threads totais não é suficiente.\n");

    cudaFree(vecA);
    cudaFree(vecB);
    cudaFree(vecV);

    return 0;
} 