#include <stdio.h>

__global__ void num_pares(){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx%2==0){
      printf("%d \n",idx);
    }
}

int main(){

    num_pares<<<2,10>>>();
    cudaDeviceSynchronize();
    return 0;
}