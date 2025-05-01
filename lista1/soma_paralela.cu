#include <stdio.h>
#include <cuda_runtime.h>

// Kernel CUDA para somar os elementos do vetor
__global__ void somaParalela(int* vetor, int* resultado, int tamanho) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cada thread soma uma parte do vetor
    int soma_local = 0;
    for (int i = tid; i < tamanho; i += stride) {
        soma_local += vetor[i];
    }
    
    // Redução dentro do bloco
    __shared__ int shared_soma[256];
    int local_tid = threadIdx.x;
    shared_soma[local_tid] = soma_local;
    __syncthreads();
    
    // Redução paralela dentro do bloco
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (local_tid < s) {
            shared_soma[local_tid] += shared_soma[local_tid + s];
        }
        __syncthreads();
    }
    
    // O primeiro thread de cada bloco escreve o resultado parcial
    if (local_tid == 0) {
        resultado[blockIdx.x] = shared_soma[0];
    }
}

int main() {
    const int tamanho = 100000;
    int* vetor = new int[tamanho];
    int* resultado = new int[1];
    
    // Inicializa o vetor
    for (int i = 0; i < tamanho; i++) {
        vetor[i] = i;
    }
    
    // Aloca memória na GPU
    int* d_vetor;
    int* d_resultado;
    cudaMalloc(&d_vetor, tamanho * sizeof(int));
    cudaMalloc(&d_resultado, sizeof(int));
    
    // Copia dados para a GPU
    cudaMemcpy(d_vetor, vetor, tamanho * sizeof(int), cudaMemcpyHostToDevice);
    
    // Configuração do kernel
    int blockSize = 256;
    int numBlocks = (tamanho + blockSize - 1) / blockSize;
    
    // Executa o kernel
    somaParalela<<<numBlocks, blockSize>>>(d_vetor, d_resultado, tamanho);
    
    // Copia o resultado de volta para a CPU
    cudaMemcpy(resultado, d_resultado, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Resultado da soma paralela: %d\n", resultado[0]);
    
    // Libera memória
    cudaFree(d_vetor);
    cudaFree(d_resultado);
    delete[] vetor;
    delete[] resultado;
    
    return 0;
} 