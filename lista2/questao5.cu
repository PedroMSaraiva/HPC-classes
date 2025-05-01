#include <stdio.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE 5000
#define NUM_BLOCKS 8
#define NUM_THREADS 16


__global__ void initVectors(float *A, float *B, float *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < VECTOR_SIZE; i += stride) {
        
        A[i] = i * 0.1f;         
        B[i] = i * 0.2f + 1.0f;  
        C[i] = i * 0.15f + 0.5f; 
    }
}

__global__ void vectorExpression(float *A, float *B, float *C, float *V, float k1, float k2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < VECTOR_SIZE; i += stride) {
        V[i] = k1 * A[i] + k2 * (B[i] + C[i]);
    }
}

int main() {
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *h_V = NULL;
    
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    float *d_V = NULL;
    
    const float k1 = 2.5f;
    const float k2 = 1.5f;
    
    size_t size = VECTOR_SIZE * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_V = (float*)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_V == NULL) {
        fprintf(stderr, "Falha na alocação de memória no host\n");
        goto cleanup;
    }
    
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocação de d_A: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocação de d_B: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocação de d_C: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_V, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocação de d_V: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    initVectors<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha no lançamento do kernel initVectors: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    vectorExpression<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_V, k1, k2);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha no lançamento do kernel vectorExpression: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na sincronização: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na cópia de d_A para h_A: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na cópia de d_B para h_B: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na cópia de d_C para h_C: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(h_V, d_V, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na cópia de d_V para h_V: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    printf("Constantes: k1 = %.2f, k2 = %.2f\n", k1, k2);
    
    printf("\nPrimeiros elementos dos vetores:\n");
    for (int i = 0; i < 5; i++) {
        printf("A[%d] = %.2f, B[%d] = %.2f, C[%d] = %.2f, V[%d] = %.2f\n",
               i, h_A[i], i, h_B[i], i, h_C[i], i, h_V[i]);
    }
    
    printf("\nÚltimos elementos dos vetores:\n");
    for (int i = VECTOR_SIZE - 5; i < VECTOR_SIZE; i++) {
        printf("A[%d] = %.2f, B[%d] = %.2f, C[%d] = %.2f, V[%d] = %.2f\n",
               i, h_A[i], i, h_B[i], i, h_C[i], i, h_V[i]);
    }
    
    printf("\nVerificação do cálculo para alguns elementos:\n");
    for (int i = 0; i < VECTOR_SIZE; i += 1000) {
        float expected = k1 * h_A[i] + k2 * (h_B[i] + h_C[i]);
        printf("V[%d] = %.2f, Esperado = %.2f, Diff = %.6f\n", 
               i, h_V[i], expected, fabsf(h_V[i] - expected));
    }
    
cleanup:
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    if (d_V) cudaFree(d_V);
    
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    if (h_C) free(h_C);
    if (h_V) free(h_V);
    
    return 0;
} 