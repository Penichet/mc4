
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;


__global__ void global_reduce_kernel(int* d_out, int* d_in, int size)
{   //indices
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    
    // do reduction in global mem
    for (unsigned int cap = blockDim.x / 2; cap > 0; cap >>= 1) 
    {
        //only compute if on lower portion of block
        if (tid < cap)
        {

            //if thread out of range or threads comp out of range, do nothing
            if(myId >= size || myId + cap >=size){
                //do nothing
            }
            else{
                // store minimum only between two valid elements in lower portion
                d_in[myId] = min(d_in[myId], d_in[myId + cap]);
            }
            
        }
        //wait for all threads to complete
        __syncthreads();
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

void reduce(int* d_out, int* d_intermediate, int* d_in, int size)
{
    /*int threads_num, numProcs;
    cudaDeviceGetAttribute(&threads_num, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    printf("max threads per mp: %d\n",  threads_num);
    cudaDeviceGetAttribute(&numProcs,cudaDevAttrMultiProcessorCount, 0);
    printf("mp count: %d\n", numProcs);*/

    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;

    //ceiling of blocks required
    int blocks = (size / maxThreadsPerBlock)+1; 
    

    global_reduce_kernel<<<blocks, threads >>>(d_intermediate, d_in, size);

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;

    // set threads to multiple of two greater than or equal to size 
    int mult = 1;
    while (mult < threads) mult *= 2;

    //launch kernel with multiple of 2 threads, and size equal to number of valid entries
    global_reduce_kernel<<<blocks, mult >>>(d_out, d_intermediate, threads);
    
}


__global__ void global_parity(int* d_out, int* d_in, int size) {
    //indices
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //idiomatic code from nvidia help
    for (int i = index; i < size; i += stride) {
        d_out[i] = d_in[i] % 10;
    }
}

void parity(int* d_out, int* d_in, int size) {
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    //ceiling of blocks required
    int blocks = (size / maxThreadsPerBlock) + 1;

    //run kernel
    global_parity <<<blocks, threads>>> (d_out, d_in, size);

    //wait for all threads to synch
    cudaDeviceSynchronize();
}


int main() {
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




    //timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //allocated device memory
    int *d_arr, *d_out, *d_intermediate;
    cudaMalloc((void**)&d_arr, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_out, sizeof(int));
    cudaMalloc((void**)&d_intermediate, arr.size() * sizeof(int));

    // treat pointer to start of vector as array pointer
    cudaMemcpy(d_arr, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    //run reduce operation
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_arr, arr.size());
    cudaEventRecord(stop, 0);
    
    //wait for it to finish
    cudaDeviceSynchronize();

    //store answer on host
    int ans;
    cudaMemcpy(&ans, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //find time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    //output to file
    ofstream outfile("q1a.txt");
    if (outfile.is_open())
    {
        outfile << ans;
        outfile.close();
    }
    //else cout << "Unable to open file";

    //print stuff
    //cout << "minimum entry found: " << ans << endl;
    //cout << "elapsted reduce time: " << elapsedTime << endl;

    //free device memory
    cudaFree(d_arr);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    //////////////////////////////////***PARITY CODE***///////////////////////////////////////

    //allocate device memory
    int* d_arrb, * d_outb;
    cudaMalloc((void**)&d_arrb, arr.size() * sizeof(int));
    cudaMalloc((void**)&d_outb, arr.size() * sizeof(int));

    //copy array A to device Memory
    cudaMemcpy(d_arrb, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    //cuda events
    cudaEvent_t b1, b2;
    cudaEventCreate(&b1);
    cudaEventCreate(&b2);


    //run parity
    cudaEventRecord(b1, 0);
    parity(d_outb, d_arrb, arr.size());
    cudaEventRecord(b2, 0);

    //calc time
    float b_time;
    cudaEventElapsedTime(&b_time, b1, b2);

    //store answer on host
    int* ans_arr = (int*)malloc(sizeof(int) * arr.size());
    cudaMemcpy(ans_arr, d_outb, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost);

    //validate
     /*for (int i = 0; i < arr.size(); i++) {
         assert(arr[i] % 10 == ans_arr[i]);
     }*/


    //output to file
    ofstream outfile2("q1b.txt");
    if (outfile2.is_open())
    {
        //avoid comma at end of string
        outfile2 << ans_arr[0];
        for (int i = 1; i < arr.size(); i++) {
            outfile2 << "," << ans_arr[i];
        }
        
        outfile2.close();
    }
    //else cout << "Unable to open file";
    
 
    //time taken output
    //cout << "Parity time taken: " << b_time << endl;

    cudaFree(d_arrb);
    cudaFree(d_outb);
    free(ans_arr);

    return 0;
}




