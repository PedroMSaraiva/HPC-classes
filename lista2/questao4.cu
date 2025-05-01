#include <stdio.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE 4000
#define NUM_BLOCKS 8
#define NUM_THREADS 512

__global__ void processVector(float *vector, float initialValue) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < VECTOR_SIZE; i += stride) {
        vector[i] = initialValue;
        
        if (i % 2 == 0) {
            vector[i] = vector[i] * vector[i];
        } else {
            vector[i] = vector[i] * vector[i] * vector[i];
        }
    }
}

bool verifyResults(float *vector, float initialValue) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float expectedValue;
        
        if (i % 2 == 0) {
            expectedValue = initialValue * initialValue;
        } else {
            expectedValue = initialValue * initialValue * initialValue;
        }
        
        const float epsilon = 1e-5;
        if (fabs(vector[i] - expectedValue) > epsilon) {
            printf("Erro no índice %d: valor = %f, esperado = %f\n", 
                   i, vector[i], expectedValue);
            return false;
        }
    }
    
    return true;
}

int main() {
    float *h_vector = NULL;  
    float *d_vector = NULL;  
    float initialValue = 3.0f;  
    

    h_vector = (float*)malloc(VECTOR_SIZE * sizeof(float));
    if (h_vector == NULL) {
        fprintf(stderr, "Falha na alocação de memória no host\n");
        return -1;
    }
    
    cudaError_t err = cudaMalloc((void**)&d_vector, VECTOR_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocação de memória na GPU: %s\n", 
                cudaGetErrorString(err));
        free(h_vector);
        return -1;
    }
    
    processVector<<<NUM_BLOCKS, NUM_THREADS>>>(d_vector, initialValue);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha no lançamento do kernel: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_vector);
        free(h_vector);
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na sincronização: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_vector);
        free(h_vector);
        return -1;
    }
    
    err = cudaMemcpy(h_vector, d_vector, VECTOR_SIZE * sizeof(float), 
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na cópia de dados da GPU para o host: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_vector);
        free(h_vector);
        return -1;
    }
    
    bool success = verifyResults(h_vector, initialValue);
    
    if (success) {
        printf("Operações realizadas com sucesso!\n");
    } else {
        printf("Falha nas operações.\n");
    }
    
    printf("\nAlguns valores do vetor processado:\n");
    for (int i = 0; i < 10; i++) {
        printf("vector[%d] = %f\n", i, h_vector[i]);
    }
    
    cudaFree(d_vector);
    free(h_vector);
    
    return 0;
} 