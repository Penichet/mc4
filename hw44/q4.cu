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

    //MAP 0s in d_in as 1 in flags array, vice versa
    if (myId < size && (mask & d_in[myId]) == 0) {
        flags[myId] = 1;
    }
    else {
        flags[myId] = 0;
    }
    //synch all threads
    __syncthreads();
}


__global__ void prescan(int* d_out, int* scan_store, int* flag, int size, bool store) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    //Blelloch scan that only works on powers of 2. this is mitigated by each block running
    //512 threads, and scan_store array is padded to powers of 2

    // load inputs into memory 
    d_out[myId] = flag[myId]; 
    __syncthreads();

    //terrible log math to index to parent and children vice versa
    //build up SUM
    for (int h = 1; h < size; h *= 2) {
        int index = h * 2 * tid + h * 2 - 1 + (size * blockIdx.x);
        if (index < (size * (blockIdx.x + 1))) { //check if myid +step is smaller than size
            d_out[index] += d_out[h * 2 * tid + h - 1 + (size * blockIdx.x)];
            //NEED TO PAD INPUT ARRAY
        }
        __syncthreads(); 
    }
    __syncthreads();
     //clear the last element
    if (tid == 0) {
        d_out[(size * (blockIdx.x+1)) - 1] = 0;
    }
    __syncthreads();

    //terrible log math to index to parent and children vice versa
    //Build down SCAN
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
    __syncthreads();
    //store the last element if thread 0 at block index
    if (tid == size -1 && store) {
        scan_store[blockIdx.x] = d_out[(size * (blockIdx.x + 1)) - 1] + flag[(size * (blockIdx.x + 1)-1)];
    }
    __syncthreads();
}

__global__ void combine(int* results, int* mini, int size, int* numFalse, int* scan) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //shared value for each block to only do once
    __shared__ int presum;
    if (threadIdx.x == 0) {
        presum = mini[blockIdx.x];
    }
    __syncthreads();
    // add total scan of each block to each element
    if (myId < size) {
        results[myId] += presum;
    }
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
    int t;
        if(myId<size)
            t = myId - scan[myId] + *numFalse; // true index for all
        __syncthreads();

        //use scan value if bit is 0, use t otherwise
        if (myId < size)
            if ((mask & d_in[myId]) == 0) d_out[scan[myId]] = d_in[myId];
            else d_out[t] = d_in[myId];
        __syncthreads();

    ////run next bit mask
}

__global__ void swap(int* d_in, int* d_out, int size) {
    //Stores each valid element from d_out to d_in, all in parallel
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    __syncthreads();
    if(myId < size) d_in[myId] = d_out[myId];
    __syncthreads();
}

int main() {
    //READING ONLY 8192 for simplicity, no padding
    /*
    TODO: Account for each thread taking care of 2 elements in prescan
    */
    vector<int> arr;
    string line;
    ifstream myfile("inp.txt");
    ofstream outfile2("q4.txt");
    if (myfile.is_open())
    {
        //gets next int
        //int numin = 0;
        while (getline(myfile, line, ','))
        {
            arr.push_back(stoi(line, nullptr));
            //arr.push_back(2); DEBUG
            //numin++;
        }
        myfile.close();
    }
    //size calculations plus padding
    int size = arr.size();
    cout << arr.size() << endl;
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = (size / maxThreadsPerBlock);
    int padblocks = 1;
    int mod = size % threads;
    if (mod > 0) blocks++;
    //padding for block cudamalloc
    while (padblocks < blocks) padblocks *= 2;
    //bootleg push 2 for everything, try to not shuffle the end
    for (int i = 0; i < threads - mod; i++) {
        arr.push_back(2);
    }

    //allocate device memory
    int* d_in, * d_out, * scan, * flags, * scan_store, * scan_large, *numFalse;
    //int* h_false = (int*)malloc(sizeof(int));
    
    //allocate memory for full size of padded blocks, only write to actual values
    cudaMalloc((void**)&flags, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_out, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan_store, padblocks * sizeof(int));
    cudaMalloc((void**)&scan_large, padblocks * sizeof(int));
    cudaMalloc((void**)&numFalse, sizeof(int));
    // treat pointer to start of vector as array pointer
    cudaMalloc((void**)&d_in, arr.size() * sizeof(int));
    cudaMemcpy(d_in, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    

    ///////////////////////////////////////////////////////////////////////////
    //For each bit 0-999
    for (int i = 0, mask = 1; i < 10; i++, mask <<= 1) {
        //MAP 0's to flags array
        global_get_flags<<<blocks, threads>>>(d_in, flags, mask, size);
        cudaDeviceSynchronize();

        //do first scan on each block and store results in scan_store array
        prescan<<<blocks, threads>>>(scan, scan_store, flags, threads, true);
        cudaDeviceSynchronize();
        //DEBUG
        /*int* h_scan = (int*)malloc(sizeof(int) * blocks);
        cudaMemcpy(h_scan, scan_store, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
        cout << "SCAN_STORE: " << h_scan[0];
        for (int j = 1; j < blocks; j++) {
            cout << "," << h_scan[j];
        }
        cout << endl;*/
        //////////////////////////

        //do secondary scan on array of scan results 
        prescan<<<1, blocks >>>(scan_large, NULL, scan_store, padblocks, false);
        cudaDeviceSynchronize();

        // Combine scans in parallel
        combine<<<blocks, threads>>>(scan, scan_large, size, numFalse, flags);
        cudaDeviceSynchronize();

        //DEBUG
        /*cudaMemcpy(h_false, numFalse, sizeof(int), cudaMemcpyDeviceToHost);
        printf("number of false: %d\n", h_false[0]);*/

        //shuffle to new values
        shuffle<<<blocks, threads>>>(d_out, d_in, scan, numFalse, size, mask);
        cudaDeviceSynchronize();

        ////move d_out to d_in to redo, must do in seperate kernel to avoid race conditions
        swap<<<blocks, threads>>>(d_in, d_out, size);
        cudaDeviceSynchronize();
    }

    /////////////////////////////////////////////////////////////////////////
    //Copy results to host from device
    int* ans_arr = (int*)malloc(sizeof(int) * (arr.size() + mod));
    cudaMemcpy(ans_arr, d_in, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost);

    // output to file   
    if (outfile2.is_open())
    {       
        //avoid comma at end of string
        outfile2 << ans_arr[0];

        //append integers up to original input size
        for (int i = 1; i < size; i++) {
            outfile2 << "," << ans_arr[i];
        }

        outfile2.close();
    }

    //free mem
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(scan);
    cudaFree(scan_store);
    cudaFree(scan_large);
    cudaFree(flags);
    cudaFree(numFalse);
    free(ans_arr);
}


