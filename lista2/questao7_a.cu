#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// Kernel para inicializar vetor em paralelo
__global__ void initializeVector(float value, float *vector, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < N; i += stride) {
        vector[i] = value;
    }
}

// Kernel para somar vetores
__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

// Função para verificar os resultados
void checkElementsAre(float target, float *vector, int N) {
    for (int i = 0; i < N; i++) {
        if (vector[i] != target) {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
            exit(1);
        }
    }
    printf("Success! All values calculated correctly.\n");
}

// Função para medir o tempo
double getElapsedTime(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

int main() {
    // Tamanho do vetor
    const int N = 2<<20; // Reduzido para testes mais rápidos
    size_t size = N * sizeof(float);
    
    // Variáveis para medir tempo
    struct timespec start_init, end_init, start_add, end_add;
    double time_init, time_add;
    
    // Obter o dispositivo atual
    int device = -1;
    cudaGetDevice(&device);
    
    // Alocar vetores na memória unificada
    float *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    
    // Configuração de execução
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Experimento A: Prefetch apenas o vetor 'a' para o dispositivo
    printf("Realizando prefetch apenas do vetor 'a' para o dispositivo...\n");
    cudaMemPrefetchAsync(a, size, device, NULL);
    
    // Inicializar vetores em paralelo
    printf("Inicializando vetores em paralelo...\n");
    clock_gettime(CLOCK_MONOTONIC, &start_init);
    
    initializeVector<<<numBlocks, blockSize>>>(3.0f, a, N);
    initializeVector<<<numBlocks, blockSize>>>(4.0f, b, N);
    initializeVector<<<numBlocks, blockSize>>>(0.0f, c, N);
    
    // Sincronizar para garantir que a inicialização está completa
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &end_init);
    time_init = getElapsedTime(start_init, end_init);
    printf("Tempo de inicialização: %.2f ms\n", time_init);
    
    // Executar o kernel de adição
    printf("Executando a adição de vetores...\n");
    clock_gettime(CLOCK_MONOTONIC, &start_add);
    
    addVectorsInto<<<numBlocks, blockSize>>>(c, a, b, N);
    
    // Verificar erros
    cudaError_t addVectorsErr = cudaGetLastError();
    if (addVectorsErr != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(addVectorsErr));
    }
    
    // Sincronizar para garantir que a adição está completa
    cudaError_t asyncErr = cudaDeviceSynchronize();
    if (asyncErr != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(asyncErr));
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_add);
    time_add = getElapsedTime(start_add, end_add);
    printf("Tempo de adição: %.2f ms\n", time_add);
    
    // Verificar resultados
    printf("Verificando resultados...\n");
    checkElementsAre(7.0f, c, N);
    
    // Liberar memória
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    printf("Tempo total: %.2f ms\n", time_init + time_add);
    
    return 0;
} 