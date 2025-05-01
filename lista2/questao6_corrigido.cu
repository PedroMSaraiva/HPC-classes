#include <stdio.h>

const int N = 16;
const int blocksize = 4;
const int SHARED_MEM_SIZE = 16;

__global__ void kernelOtimizado(int *data, int *outdata, int N) {
    __shared__ int sharedData[SHARED_MEM_SIZE];
    
    int tid = threadIdx.x;
    int ix = blockIdx.x * blockDim.x + tid;
    
    sharedData[tid] = data[ix];
    __syncthreads();
    
    int p = 0;
    
    for (int i = 0; i < N; i++) {
        if (data[ix] > data[i]) {
            p++;
        }
    }
    
    outdata[p] = data[ix];
}


__global__ void kernelEficiente(int *data, int *outdata, int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < N) {
        int p = 0;
        
        for (int i = 0; i < N; i++) {
            p += (data[ix] > data[i]) ? 1 : 0;
        }
        
        outdata[p] = data[ix];
    }
}

int main() {
    int a[N] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    int b[N] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int isize = N * sizeof(int);
    int *ad, *bd;
    
    cudaError_t err = cudaMalloc((void**)&ad, isize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na alocação de memória para ad: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&bd, isize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na alocação de memória para bd: %s\n", cudaGetErrorString(err));
        cudaFree(ad);
        return -1;
    }
    
    err = cudaMemcpy(ad, a, isize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na cópia de dados para GPU: %s\n", cudaGetErrorString(err));
        cudaFree(ad);
        cudaFree(bd);
        return -1;
    }
    
    kernelEficiente<<<(N + blocksize - 1) / blocksize, blocksize>>>(ad, bd, N);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro no lançamento do kernel: %s\n", cudaGetErrorString(err));
        cudaFree(ad);
        cudaFree(bd);
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na sincronização: %s\n", cudaGetErrorString(err));
        cudaFree(ad);
        cudaFree(bd);
        return -1;
    }
    
    err = cudaMemcpy(b, bd, isize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro na cópia de dados para o host: %s\n", cudaGetErrorString(err));
        cudaFree(ad);
        cudaFree(bd);
        return -1;
    }
    
    cudaFree(ad);
    cudaFree(bd);
    
    printf("Vetor ordenado: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
    
    bool ordenado = true;
    for (int i = 1; i < N; i++) {
        if (b[i-1] > b[i]) {
            ordenado = false;
            break;
        }
    }
    
    if (ordenado) {
        printf("Ordenação realizada com sucesso!\n");
    } else {
        printf("Falha na ordenação.\n");
    }
    
    return 0;
}

/* 
Algoritmo original:
- Complexidade: O(N²)
- Problemas: Lógica incorreta, sem verificação de erros, uso de memória ineficiente
- Não realiza a ordenação corretamente

Algoritmo otimizado:
- Complexidade: Ainda O(N²), mas com melhor uso da memória e correção da lógica
- Melhorias: Adiciona verificação de erros, corrige lógica de ordenação, usa a contagem para posicionamento
- Opções alternativas: 
  1. kernelOtimizado: usa memória compartilhada
  2. kernelEficiente: implementa acessos mais coalescidos e evita divergência

Para conjuntos de dados maiores, seria recomendável:
1. Usar algoritmos de ordenação paralelos mais eficientes (Bitonic Sort, Radix Sort)
2. Implementar estratégias de tiling para melhorar localidade
3. Utilizar bibliotecas otimizadas como Thrust ou CUB
*/ 