#include <stdio.h>
#include <stdlib.h>
#define TPB 512

__global__ void countGlobal(int *a, int *b, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        atomicAdd(&b[a[index]/100], 1);
    }
}

__global__ void countLocal(int *a, int *b, int n){
    __shared__ int B[10];
    int i;

    if(threadIdx.x < 10){
        B[threadIdx.x] = 0;
    }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        atomicAdd(&B[a[index]/100], 1);
    }

    __syncthreads();

    if(threadIdx.x < 10){
        atomicAdd(&b[threadIdx.x], B[threadIdx.x]);
    }

}

__global__ void upwardSweep(int *temp, int *c, int d, int n){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n){
        if(index >= d){
            temp[index] = c[index-d];
        }
        else{
            temp[index] = 0;
        }

    }
    
}  

__global__ void downwardSweep(int *temp, int *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n){
        c[index] = c[index] + temp[index];
    }
}

int main(int argc, char ** argv){

    FILE *fp;
    char word[255];
    int N = 0;

    fp = fopen("inp.txt", "r");

    if (fp == NULL){
        return -1;
    }


    while (fscanf(fp, "%*s ", word) != EOF) {     //this one works      
        N++;
    }

    if(N == 0){
        return -1;
    } 

    fclose(fp);

    fp = fopen("inp.txt", "r");

    if (fp == NULL){
        return -1;
    }

    int A[N];

    fscanf(fp, "%d", &A[0]);

    int idx = 1;
    while(idx < N){
        fscanf(fp, ", %d", &A[idx]);
        idx++;
    }

    int B[10];

    for(int i = 0; i < 10; i++){
        B[i] = 0;
    }

    int *d_A;
    int *d_B;

    int A_size = sizeof(int)*N;
    int B_size = sizeof(int)*10;

    cudaMalloc((void **) &d_A, A_size);
    cudaMalloc((void **) &d_B, B_size);

    cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice);

    countGlobal<<<(N + TPB - 1)/TPB, TPB>>>(d_A, d_B, N);

    cudaMemcpy(B, d_B, B_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    fp = fopen("q2a.txt","w+");
    fputs("[",fp);
    for(int i = 0; i < 10; i++){
        snprintf(word, sizeof(word), "%d", B[i]); // convert int to char array
        fputs(word,fp);

        if(i != 9) // comma after every digit except last
            fputs(", ",fp);
    }
    fputs("]",fp);
    fclose(fp);

    for(int i = 0; i < 10; i++){
        B[i] = 0;
    }

    // part b
    cudaMalloc((void **) &d_A, A_size);
    cudaMalloc((void **) &d_B, B_size);

    cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice);

    countLocal<<<(N + TPB - 1)/TPB, TPB>>>(d_A, d_B, N);

    cudaMemcpy(B, d_B, B_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    fp = fopen("q2b.txt","w+");
    fputs("[",fp);
    for(int i = 0; i < 10; i++){
        snprintf(word, sizeof(word), "%d", B[i]); // convert int to char array
        fputs(word,fp);

        if(i != 9) // comma after every digit except last
            fputs(", ",fp);
    }
    fputs("]",fp);
    fclose(fp);

    // part c

    int C[10];
    int temp[10];

    for(int i = 0; i < 10; i++){
        C[i] = B[i];
    }

    int *d_C;
    int C_size = 10*sizeof(int);
    int temp_size = 10*sizeof(int);
    int d = 1;

    cudaMalloc((void **) &d_temp, temp_size);
    cudaMalloc((void **) &d_C, C_size);

    cudaMemcpy(d_C, C, C_size, cudaMemcpyHostToDevice);

    while(d < 10) {

        upwardSweep<<<(10 + TPB - 1)/TPB, TPB>>>(d_temp, d_C, d, 10);
    
        downwardSweep<<<(10 + TPB - 1)/TPB, TPB>>>(d_temp, d_C, 10);
        
        d *= 2;

    }

    cudaMemcpy(C, d_C, C_size, cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_C);

    fp = fopen("q2c.txt","w+");
    fputs("[",fp);
    for(int i = 0; i < 10; i++){
        snprintf(word, sizeof(word), "%d", C[i]); // convert int to char array
        fputs(word,fp);

        if(i != 9) // comma after every digit except last
            fputs(", ",fp);
    }
    fputs("]",fp;
    fclose(fp);

    return 0;

}