#include <stdio.h>


__global__ void soma_vetores(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        c[idx] = a[idx] + b[idx];
        idx += blockDim.x * gridDim.x;
    }
}


void verificacao(int *c, int N) {
    for(int i = 0; i < N; i++) {
        if(c[i] != c[0]) {
            printf("Erro detectado no codigo.\n");
            exit(1);
        }
    }
    printf("Sucesso! Todos os valores sao identicos.\n");
}

int main(void) {
    const int N = 1000;
    
    int h_a[1000], h_b[1000], h_c[1000];
    
    // Inicializa os dados no host
    // a: crescente (0 a N-1)
    // b: decrescente (N-1 a 0)
    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = N - 1 - i;
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    soma_vetores<<<8, 16>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Primeiros 10 elementos:\n");
    printf("A: ");
    for(int i = 0; i < 10; i++) printf("%d ", h_a[i]);
    printf("\nB: ");
    for(int i = 0; i < 10; i++) printf("%d ", h_b[i]);
    printf("\nC: ");
    for(int i = 0; i < 10; i++) printf("%d ", h_c[i]);
    printf("\n\n");
    
    verificacao(h_c, N);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
} 