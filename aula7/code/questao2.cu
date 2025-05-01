#include <stdio.h>

__global__ void init_vetor(int *vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        vec[idx] = 100;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void incrementa_vetor(int *vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        vec[idx] += idx;
        idx += blockDim.x * gridDim.x;
    }
}

int main(void) {
    const int N = 100;
    int h_vec[100];
    int *d_vec;
    
    for(int i = 0; i < N; i++) {
        h_vec[i] = 100;
    }
    
    cudaMalloc((void**)&d_vec, N * sizeof(int));
    cudaMemcpy(d_vec, h_vec, N * sizeof(int), cudaMemcpyHostToDevice);
    
    init_vetor<<<8, 16>>>(d_vec, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_vec, d_vec, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Após inicialização com 100:\n");
    for(int i = 0; i < N; i++) {
        printf("%d ", h_vec[i]);
        if((i + 1) % 10 == 0) printf("\n");
    }
    
    incrementa_vetor<<<8, 16>>>(d_vec, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_vec, d_vec, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nApós incremento com índices:\n");
    for(int i = 0; i < N; i++) {
        printf("%d ", h_vec[i]);
        if((i + 1) % 10 == 0) printf("\n");
    }
    
    int correto = 1;
    for(int i = 0; i < N; i++) {
        if(h_vec[i] != (100 + i)) {
            printf("\nErro na posição %d: esperado %d, encontrado %d\n", 
                   i, 100 + i, h_vec[i]);
            correto = 0;
            break;
        }
    }
    
    if(correto) {
        printf("\nSucesso! Todos os valores estão corretos.\n");
    }
    
    cudaFree(d_vec);
    return 0;
} 