#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initializeVector(int *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = idx;
    }
}
__global__ void squareVector(int *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = vec[idx] * vec[idx];
    }
}

bool verifyResults(int *vec, int size) {
    for (int i = 0; i < size; i++) {
        if (vec[i] != i * i) {
            printf("Erro na posição %d: Esperado %d, Obtido %d\n", i, i*i, vec[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1000;
    int *vec;
    size_t size = N * sizeof(int);
    cudaError_t err;

    err = cudaMallocManaged(&vec, size);
    if (err != cudaSuccess) {
        printf("Erro na alocação: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = 32;

    initializeVector<<<blocksPerGrid, threadsPerBlock>>>(vec, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Erro na inicialização: %s\n", cudaGetErrorString(err));
        cudaFree(vec);
        return 1;
    }

    squareVector<<<blocksPerGrid, threadsPerBlock>>>(vec, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Erro na operação de quadrado: %s\n", cudaGetErrorString(err));
        cudaFree(vec);
        return 1;
    }

    if (verifyResults(vec, N)) {
        printf("Sucesso! Todos os elementos foram processados corretamente.\n");
    } else {
        printf("Falha! Erros encontrados nos resultados.\n");
    }

    cudaFree(vec);
    return 0;
} 