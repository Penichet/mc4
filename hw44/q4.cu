
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
__forceinline __device__ void prefix_scan(int* d_out, int* flag, int* temp,  int size) {
    //TODO: need to adjust size to multiple of 2 and also resize temp array 

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //need to initialize all in temp to 0?
    
    d_out[myId] = flag[myId]; // load inputs into memory , maybe need to do 2 elements?

    //for h = 0 to(log n) - 1 do
    //    for all i from 0 to n - 1 in step of 2 ^ (h + 1) in parallel do
    //        B[i + 2 ^ (h + 1) - 1] = B[i + 2 ^ (h)-1] + B[i + 2 ^ (h + 1) - 1]; // Parent = sum of children
    //       temp(myId + h*2 -1) = temp(myId + h -1) + temp (myId + h*2 -1); - math replacement for log
    //int step = 2;
    for (int h = 1; h < size; h *= 2) {
        int index = 2* myId + h * 2 - 1;
        if(index < size) { //check if myid +step is smaller than size
            d_out[index] += d_out[2* myId + h - 1];
            // 1,2,3,4,5,0,0,0
            // h = 1, out(2i + 1) += out(2i); B:1,3,3,7,5,5,0,0
            // h = 2, out(2i +3) += out(2i + 1) B:1,3,3,10,5,5,0,5 
            // h = 4, out(2i + 7) += out(2i + 3) B:1,3,3,10,5,5,0,15 
            // h = 8, break
            //this is right -- NEED TO PAD INPUT ARRAY
        }
        __syncthreads();
    }

    //B[n - 1] = 0;
    if (myId == 0) {
        d_out[size - 1] = 0;// B:1,3,3,10,5,5,0,0
    } // clear the last element
    __syncthreads();

    //for h = (log n)-1 down to 0 do
    //    for all i from 0 to n-1 in steps of 2^(h+1) in parallel do
    //        hlog  = 3,2,1,0 hreal = 8,4,2,1, Good:3,2,1,0

    //        LeftVal = B[i + 2^(h) - 1]; // Save Old Left before modifying
    //        leftval = B[i + h -1]--------------------------

    //        B[i + 2^(h) - 1] = B[i + 2^(h+1) - 1]; // Left child = Parent Node
    //        B[i + (h) - 1] = B[i + (h*2) -1] ---------------------------

    //        B[i + 2^(h+1) - 1] = B[i + 2^(h+1) - 1] + LeftVal; // Right = Parent Node + Old Left
    //        B[i + (h*2) - 1] = B[i + (h*2) -1] + leftVal; ------------------------------


    //        ending at >0 to assure we dont reach case where 2^0 is equiv to 0*2
    
    // h = log(size) - 1, basically want to do log(size) iterations 
    //so start with size/4 to 0 dividing by 2 each time
    
    for (int h = size/2; h > 0; h /= 2) { //B:1,3,3,10,5,5,0,0 --- might need to do 2*myId still
        int index = h*2*myId + (h * 2) - 1; // 4 + 1 = 3
        int right = h*2*myId + h - 1;       // 4 + 0 = 2
        if (index < size) { 
            int leftVal = d_out[right];
            d_out[right] = d_out[index];
            d_out[index] += leftVal;
            //h = 4->0 left = d[3]; d[3] = d[7]; d[7] += left;    left:10   B:1,3,3,0,5,5,0,10 -- only one executes
            //h = 2->0 left = d[1]; d[1] = d[3]; d[3] += left;    left:3    B:1,0,3,3,5,5,0,10
            //     ->1 left = d[5]; d[5] = d[7]; d[7] += left;    left:5    B:1,0,3,3,5,10,0,15 -- think thats right
            //h = 1->0 left = d[0]; d[0] = d[1]; d[1] += left;    left:1    B:0,1,3,3,5,10,0,15
            //     ->1 left = d[2]; d[2] = d[3]; d[3] += left;    left:3    B:0,1,3,6,5,10,0,15
            //     ->2 left = d[4]; d[4] = d[5]; d[5] += left;    left:5    B:0,1,3,6,10,15,0,15
            //     ->3                                                      B:0,1,3,6,10,15,15,15
            //h = 0, fails out
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
            prefix_scan(scan, flags, temp, size);
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


void bucket(int* d_out, int* d_in, int* flags, int* scan, int* temp, int size) {
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = (size / maxThreadsPerBlock) + 1;
    global_bucket_sort <<<blocks, threads>>> (d_out, d_in, flags, scan, temp, size);
}

int main()
{
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
    int* d_arr, *d_out, *scan, *flags, *temp;
    cudaMalloc((void**)&d_arr, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_out, arr.size() * sizeof(int));
    cudaMalloc((void**)&flags, arr.size() * sizeof(int));
    cudaMalloc((void**)&scan, arr.size() * sizeof(int));
    cudaMalloc((void**)&temp, arr.size() * sizeof(int));


    // treat pointer to start of vector as array pointer
    cudaMemcpy(d_arr, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    //Cuda Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //run reduce operation
    cudaEventRecord(start, 0);
    bucket(d_out, d_arr, flags, scan, temp, arr.size());
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