
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

//https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//child kernel for prefix scan - NOT SURE IF INCLUSIVE OR EXCLUSIVE
//__device__ void prefix_scan(int* d_out, int* flag, int* temp,  int size) {
__global__ void prefix_scan(int* d_out, int* flag, int size) {
    //TODO: need to adjust size to multiple of 2 and also resize temp array 

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //need to initialize all in temp to 0?
    
    d_out[myId] = flag[myId]; // load inputs into memory 

    for (int h = 1; h < size; h *= 2) {
        int index = h*2* myId + h * 2 - 1;
        if(index < size) { //check if myid +step is smaller than size
            d_out[index] += d_out[h*2* myId + h - 1];

            //NEED TO PAD INPUT ARRAY
        }
        __syncthreads();
    }

    //B[n - 1] = 0;
    if (myId == 0) {
        d_out[size - 1] = 0;
    } // clear the last element
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



__global__ void global_bucket_sort(int* d_out, int* d_in, int* flags, int* scan,int* temp, int size) {
    //indices
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int numFalse;
    //int tid = threadIdx.x;
    int mask = 1;
    
    if (myId < size) {
        // for every bit, increase mask 1 bit 
        for (int i = 0; i < 10; i++, mask <<= 1) {

            //MAP 0s as 1 in flags array
            if ((mask & d_in[myId])==0) {
                //if (tid == 0) printf("masking %d with %d, returned 0", d_in[myId], mask);
                flags[myId] = 1;
            }
            //synch all threads
            __syncthreads();
            printf("we haven't failed yet");
            //run prefix scan on 0s - should be inlined??
            //prefix_scan(scan, flags, size);
            numFalse = scan[size - 1] + flags[size - 1]; // number of falses total
            __syncthreads();
            //scan now holds results of scan from flags

            int t = myId - scan[myId] + numFalse; // true index for all
            __syncthreads();

            //if bit true, use t index
            if (mask & d_in[myId] != 0) {
                d_out[t] = d_in[myId];
            }
            else { //if bit false, use scan index aka f index
                d_out[scan[myId]] = d_in[myId];
            }
            __syncthreads();

            //copy to d_in to redo next layer
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
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = (size / maxThreadsPerBlock) + 1;
    //global_bucket_sort <<<blocks, threads>>> (d_out, d_in, flags, scan, temp, size);

    // testing scan///////////////////////////////////////////
    int* temp;
    vector<int> temparr;
    for (int i = 1; i <= 8192; i++) {
        temparr.push_back(1);
    }
    cudaMalloc((void**)&temp, temparr.size() * sizeof(int));
    cudaMemcpy(temp, &temparr[0], temparr.size() * sizeof(int), cudaMemcpyHostToDevice);
    printf("starting prefix scan\n");
    // more than 1024 is messed up, prolly bc 1024 threads per block
    int tempsize = (8192 / 512);
    prefix_scan <<<tempsize, 512 >>> (scan, temp, temparr.size());
    int* ans_scan = (int*)malloc(sizeof(int) * temparr.size());
    cudaMemcpy(ans_scan, scan, sizeof(int) * temparr.size(), cudaMemcpyDeviceToHost);
    cout << ans_scan[0];
    for (int i = 1; i < 8192; i++) {
        cout << "," << ans_scan[i];
    }
    ///////////////////////////////

}

int main(){
    vector<int> arr;
    string line;
    ifstream myfile("inp.txt");
    if (myfile.is_open())
    {
        //gets next int
        while (getline(myfile, line, ','))
        {
            arr.push_back(stoi(line, nullptr));
        }
        myfile.close();
    }
    else cout << "Unable to open file";
    //Array A is now accessible as arr

    //allocate device memory
    int* d_arr, *d_out, *scan, *flags;
    cudaMalloc((void**)&d_arr, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_out, arr.size() * sizeof(int));
    cudaMalloc((void**)&flags, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan, arr.size() * sizeof(int));
    //cudaMalloc((void**)&temp, arr.size() * sizeof(int));


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

    //output to file
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