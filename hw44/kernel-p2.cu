#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;


__global__ void global_get_flags(int* d_in, int* flags, int mask, int size) {
    //indices
    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    if (myId < size) {
        //MAP 0s as 1 in flags array
        if ((mask & d_in[myId]) == 0) {
            flags[myId] = 1;
        }
        else {
            flags[myId] = 0;
        }
        //synch all threads
        __syncthreads();
    }
}


__global__ void prescan(int* d_out, int* scan_store, int* flag, int size, bool store) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    //size is num of threads, but need to check compared to size * block -- ACCOUNT FOR 2x thread 

    d_out[myId] = flag[myId]; // load inputs into memory 
    __syncthreads();

    for (int h = 1; h < size; h *= 2) {
        int index = h * 2 * tid + h * 2 - 1 + (size * blockIdx.x);
        if (index < (size * (blockIdx.x + 1))) { //check if myid +step is smaller than size
            d_out[index] += d_out[h * 2 * tid + h - 1 + (size * blockIdx.x)];
            //NEED TO PAD INPUT ARRAY
        }
        __syncthreads(); 
    }

     //clear the last element
    if (tid == 0) {
        d_out[(size * (blockIdx.x+1)) - 1] = 0;
    }
    __syncthreads();

    for (int h = size / 2; h > 0; h /= 2) {
        int index = h * 2 * tid + (h * 2) - 1 + (size * blockIdx.x);
        int right = h * 2 * tid + (h * 1) - 1 + (size * blockIdx.x);
        if (index < (size * (blockIdx.x+1))) {
            int leftVal = d_out[right];
            d_out[right] = d_out[index];
            d_out[index] += leftVal;
        }
        __syncthreads();
    }

    //store the last element if thread 0 at block index
    if (tid == 0 && store) {
        scan_store[blockIdx.x] = d_out[(size * (blockIdx.x + 1)) - 1] + flag[(size * (blockIdx.x + 1)-1)];
    }
    __syncthreads();
}

__global__ void combine(int* d_out, int* results, int* mini, int size, int* numFalse, int* scan) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int presum;
    if (threadIdx.x == 0) {
        presum = mini[blockIdx.x];
    }
    __syncthreads();
    results[myId] += presum;
    __syncthreads();
    // store largest num in numFalse pointer location
    if (myId == size -1 ) {
        *numFalse = results[size - 1] + scan[size -1];
        //printf("num false cuda: %d\n", d_out[size - 1]);
    }
    __syncthreads();

}

__global__ void shuffle(int* d_out, int* d_in, int* scan, int* numFalse, int size, int mask) {
    //overlap at 1015 at end of cycle with storing indices
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int t = myId - scan[myId] + *numFalse; // true index for all
    __syncthreads();

    /*if ((mask & d_in[myId]) == 0) d_out[myId] = scan[myId];
    else d_out[myId] = t; 
    __syncthreads();*/
    //use scan value if bit is 0, use t otherwise
    if ((mask & d_in[myId]) == 0) d_out[scan[myId]] = d_in[myId];
    else d_out[t] = d_in[myId];
    __syncthreads();
    
    //////copy to d_in to redo next layer
    //d_in[myId] = d_out[myId];
    //__syncthreads();

    ////clear used arrays
    scan[myId] = 0;
    __syncthreads();
    
    ////run next bit mask
}

__global__ void swap(int* d_in, int* d_out, int size) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    __syncthreads();
    if(myId < size) d_in[myId] = d_out[myId];
    __syncthreads();
}

int main() {
    //READING ONLY 8192 for simplicity, no padding
    /*
    TODO: Pad d_in for final block
          Pad scan_store to power of 2
          Account for each thread taking care of 2 elements in prescan
          Free cuda memory after
          Account for myId > size in new functions 
          Figure out why it works with manual iterations
    */
    int size = 8192;
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = (size / maxThreadsPerBlock);
    if (size % threads != 0) {
        blocks++;
    }
    vector<int> arr;
    string line;
    int mask = 1 << 0;
    ifstream myfile("inp.txt");
    ofstream outfile2("testfull.txt");
    if (myfile.is_open())
    {
        //gets next int
        int numin = 0;
        while (getline(myfile, line, ',') && numin < size)
        {
            arr.push_back(stoi(line, nullptr));
            //arr.push_back(2);
            numin++;
        }
        myfile.close();
    }

    //allocate device memory
    int* d_in, * d_out, * scan, * flags, * scan_store, * scan_large, *numFalse;
    int* h_false = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_out, arr.size() * sizeof(int));
    cudaMalloc((void**)&flags, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan_store, blocks * sizeof(int));
    cudaMalloc((void**)&scan_large, blocks * sizeof(int));
    cudaMalloc((void**)&numFalse, sizeof(int));
    // treat pointer to start of vector as array pointer
    cudaMalloc((void**)&d_in, arr.size() * sizeof(int));
    cudaMemcpy(d_in, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    

    ///////////////////////////////////////////////////////////////////////////
    
    
    for (int i = 0; i < 10; i++, mask <<= 1) {
        //MAP 0's to flags array
        global_get_flags << <blocks, threads >> > (d_in, flags, mask, size);
        cudaDeviceSynchronize();
        //do scan on each block and store results in scan_store - set size to threads/2 to run on a single block
        prescan <<<blocks, threads >>> (scan, scan_store, flags, threads, true);
        cudaDeviceSynchronize();
        //////////////////////////
        int* h_scan = (int*)malloc(sizeof(int) * blocks);
        cudaMemcpy(h_scan, scan_store, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
        cout << "SCAN_STORE: " << h_scan[0];
        for (int j = 1; j < blocks; j++) {
            cout << "," << h_scan[j];
        }
        cout << endl;
        //////////////////////////
        //do scan on array of scan results - somehow only on small
        prescan << <1, blocks >> > (scan_large, NULL, scan_store, blocks, false);
        cudaDeviceSynchronize();
        //works for 8192 and 16 so far. probably because powers of 2
        combine << <blocks, threads >> > (d_out, scan, scan_large, size, numFalse, flags);
        cudaDeviceSynchronize();
        // also works for 8192
        cudaMemcpy(h_false, numFalse, sizeof(int), cudaMemcpyDeviceToHost);
        printf("number of false: %d\n", h_false[0]); //NUm false works correctly
        //shuffle
        shuffle << <blocks, threads >> > (d_out, d_in, scan, numFalse, size, mask);
        cudaDeviceSynchronize();
        swap << <blocks, threads >> > (d_in, d_out, size);
        cudaDeviceSynchronize();
    }

    /////////////////////////////////////////////////////////////////////////

    //copy results
    //int* ans_arr = (int*)malloc(sizeof(int) * arr.size());
    int* ans_arr = (int*)malloc(sizeof(int) * arr.size());
    cudaMemcpy(ans_arr, d_in, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost);
   

    // output to file
    
    if (outfile2.is_open())
    {
        //avoid comma at end of string
       
        //avoid comma at end of string
        outfile2 << ans_arr[0];
        for (int i = 1; i < size; i++) {
            outfile2 << "," << ans_arr[i];
        }

        outfile2.close();
    }
}


