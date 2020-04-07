#include <stdio.h>
#include <stdlib.h>
#define TPB 512

__global__ void oddPopulate(int *A, int *O, int n, int *numOdds){
    __shared__ int odds;

    if(threadIdx.x){
        odds = 0;
    }

    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
       if(A[index] % 2 == 0){
           O[index] = 0;
       }
       else{
           O[index] = 1;
           atomicAdd(&odds, 1);
       }
    }

    if(threadIdx.x == 0){
        atomicAdd(numOdds, odds);
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

__global__ void fillOddArray(int *D, int *sums, int *A, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        if(A[index]%2 == 1){
            D[sums[index] - 1] = A[index];
        }
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

    int odds[1];

    int O[N];

    int *d_A;
    int *d_O;
    int *d_numOdds;

    int A_size = sizeof(int)*N;
    int O_size = sizeof(int)*N;
    int odds_size = sizeof(int)*1;

    cudaMalloc((void **) &d_A, A_size);
    cudaMalloc((void **) &d_O, O_size);
    cudaMalloc((void **) &d_numOdds, odds_size);

    cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numOdds, odds, odds_size, cudaMemcpyHostToDevice);

    oddPopulate<<<(N + TPB - 1)/TPB, TPB>>>(d_A, d_O, N, d_numOdds);

    cudaMemcpy(O, d_O, O_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(odds, d_numOdds, odds_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_O);
    cudaFree(d_numOdds);

    int *d_OSum;
    int *d_temp;
    int temp_size = N*sizeof(int);

    cudaMalloc((void **) &d_OSum, temp_size);
    cudaMalloc((void **) &d_temp, temp_size);

    cudaMemcpy(d_OSum, O, temp_size, cudaMemcpyHostToDevice);

    int x = 1;
    
    while(x < N) {

        upwardSweep<<<(N + TPB - 1)/TPB, TPB>>>(d_temp, d_OSum, x, N);
    
        downwardSweep<<<(N + TPB - 1)/TPB, TPB>>>(d_temp, d_OSum, N);
        
        x *= 2;

    }

    cudaFree(d_temp);

    int D[odds[0]];

    int *d_D;
    int *d_Nums;

    cudaMalloc((void **) &d_D, odds[0]*sizeof(int));
    cudaMalloc((void **) &d_Nums, N*sizeof(int));

    cudaMemcpy(d_Nums, A, N*sizeof(int), cudaMemcpyHostToDevice);

    fillOddArray<<<(N + TPB - 1)/TPB, TPB>>>(d_D, d_OSum, d_Nums, N);

    cudaMemcpy(D, d_D, odds[0]*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_Nums);
    cudaFree(d_D);
    cudaFree(d_OSum);

    fp = fopen("q3.txt","w+");
    fputs("[",fp);
    for(int i = 0; i < odds[0]; i++){
        snprintf(word, sizeof(word), "%d", D[i]);
        fputs(word,fp);

        if(i != odds[0] - 1)
            fputs(", ",fp);
    }
    fputs("]",fp);
    fclose(fp);

    return 0;
}

    