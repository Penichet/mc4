
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

//https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//child kernel for prefix scan - EXCLUSIVE

__forceinline __device__ void prefix_scan(int* d_out, int* flag, int size) {
//__global__ void prefix_scan(int* d_out, int* flag, int size) {
    //TODO: need to adjust size to multiple of 2 and also resize temp array 
    //__device__ int threadcount;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //need to initialize all in temp to 0?
    
    d_out[myId] = flag[myId]; // load inputs into memory 

    for (int h = 1; h < size; h *= 2) {
        int index = h * 2 * myId + h * 2 - 1;
        if (index < size) { //check if myid +step is smaller than size
            d_out[index] += d_out[h * 2 * myId + h - 1];
            //NEED TO PAD INPUT ARRAY
        }
        __syncthreads(); // only synchs a given block - NEED TO SYNC ACROSS BLOCKS SOMEHOW, ATOMIC INC??
    }
   /* if (threadIdx.x == 0) {
        atomicInc(&count, gridDim.x);
    }*/

    //clear the last element
    if (myId == 0) {
        d_out[size - 1] = 0;
    }
    __syncthreads();

    //        leftval = B[i + h -1]--------------------------
    //        B[i + (h) - 1] = B[i + (h*2) -1] ---------------------------
    //        B[i + (h*2) - 1] = B[i + (h*2) -1] + leftVal; ------------------------------
    //0,1,3,6,10,15,21,28, 36,45,55,66,78,91,105,120 - EXPECTED
                                          
    for (int h = size/2; h > 0; h /= 2) {
        int index = h*2*myId + (h * 2) - 1; 
        int right = h*2*myId + (h*1) - 1;
        if (index < size) { 
            int leftVal = d_out[right];
            d_out[right] = d_out[index];
            d_out[index] += leftVal;
        }
        __syncthreads();
    }
}

__device__ int numFalse;
__device__ unsigned int threadcount;
__device__ bool ready;

__global__ void global_bucket_sort(int* d_out, int* d_in, int* flags, int* scan, int size) {
    //indices
    ready = false;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    
    //int tid = threadIdx.x;
    int mask = 1;
    
    if (myId < size) {
        // for every bit, increase mask 1 bit 
        //for (int i = 0; i < 10; i++, mask <<= 1) {
        for (int i = 0; i < 10; i++, mask <<= 1) {
            //MAP 0s as 1 in flags array
            if ((mask & d_in[myId]) == 0) {
                flags[myId] = 1;
            }
            //FLAGS WORKS JUST FINE

            //synch all threads
            __syncthreads();

            prefix_scan(scan, flags, size);
            
            // up to here works perfectly
            
            //half the threads pass sync barrier and print numfalse as 514 first
            numFalse = scan[size - 1] + flags[size - 1]; // number of falses total
            __syncthreads();
            
            //make sure somehow all threads wait to have most up to date numFalse
            //if (atomicInc(&threadcount, 1) == 10) {
            //    numFalse = scan[size - 1] + flags[size - 1];
            //    ready = true;
            //}
            //while (!ready) {
            //    //do nothing until all threads ready
            //}
            __syncthreads();
            //printf("numFalse: %d\n", numFalse);
            

            int t = myId - scan[myId] + numFalse; // true index for all
            __syncthreads();

            //use scan value if bit is 0, use t otherwise
            if ((mask & d_in[myId]) == 0) d_out[scan[myId]] = d_in[myId];
            else d_out[t] = d_in[myId];
            __syncthreads();

            ////copy to d_in to redo next layer
            d_in[myId] = d_out[myId];
            __syncthreads();

            //clear used arrays
            flags[myId] = 0;
            scan[myId] = 0;
            __syncthreads();
            //run next bit mask
        }
    }
}


void bucket(int* d_out, int* d_in, int* flags, int* scan, int size) {
   
    //size over 10000 bugs out????
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = (size / maxThreadsPerBlock);
    if (size % threads != 0) {
        blocks++;
    }
    //more than 1024 is messed up, prolly bc 1024 threads per block
    global_bucket_sort <<<1, 512>>> (d_out, d_in, flags, scan, size);


    //prefix_scan <<<blocks, threads >>> (scan, temp, temparr.size());
    /*int* ans_scan = (int*)malloc(sizeof(int) * temparr.size());
    cudaMemcpy(ans_scan, d_out, sizeof(int) * temparr.size(), cudaMemcpyDeviceToHost);
    cout << ans_scan[0];
    for (int i = 1; i < size; i++) {
        cout << "," << ans_scan[i];
    }*/
    /////////////////////////////

}

int main(){
    //READING ONLY 8192 for simplicity, no padding
    int size = 512;
    vector<int> arr;
    string line;
    ifstream myfile("inp.txt");
    if (myfile.is_open())
    {
        //gets next int
        int numin = 0;
        while (getline(myfile, line, ',') && numin<size)
        {
            arr.push_back(stoi(line, nullptr));
            numin++;
        }
        myfile.close();
    }
    else cout << "Unable to open file";
    //Array A is now accessible as arr
    printf("size of arr: %d", arr.size());
    //allocate device memory
    int* d_arr, *d_out, *scan, *flags;
    cudaMalloc((void**)&d_arr, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_out, arr.size() * sizeof(int));
    cudaMalloc((void**)&flags, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan, arr.size() * sizeof(int));

    // treat pointer to start of vector as array pointer
    cudaMemcpy(d_arr, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    //Cuda Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //run reduce operation
    cudaEventRecord(start, 0);
    bucket(d_out, d_arr, flags, scan, arr.size());
    cudaEventRecord(stop, 0);

    //copy results
    int* ans_arr = (int*)malloc(sizeof(int) * arr.size());
    cudaMemcpy(ans_arr, d_out, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost);
    //cudaMemcpy(ans_arr, flags, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost);

   // output to file
    ofstream outfile2("q4.txt");
    if (outfile2.is_open())
    {
        //avoid comma at end of string
        outfile2 << ans_arr[0];
        for (int i = 1; i < arr.size(); i++) {
            outfile2 << "," << ans_arr[i];
        }

        outfile2.close();
    }
}