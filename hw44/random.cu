#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std; 
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

__global__ void prescan(float* g_odata, float* g_idata, int n) {
        extern __shared__ float temp[];  // allocated on invocation
        int thid = threadIdx.x;
        int offset = 1; 
        int ai = thid;
        int bi = thid + (n / 2);
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        temp[ai + bankOffsetA] = g_idata[ai];
        temp[bi + bankOffsetB] = g_idata[bi];

        for (int d = n >> 1; d > 0; d >>= 1){                    // build sum in place up the tree 
            __syncthreads();    
            if (thid < d){ 
                temp[ai + bankOffsetA] = g_idata[ai];
                temp[bi + bankOffsetB] = g_idata[bi];
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }
        if (thid == 0) { 
            temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        } //clear last element

        for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
            offset >>= 1;
            __syncthreads();
            if (thid < d){ 
                temp[ai + bankOffsetA] = g_idata[ai];
                temp[bi + bankOffsetB] = g_idata[bi];
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();

        //write results to device memory
        g_odata[ai] = temp[ai + bankOffsetA];
        g_odata[bi] = temp[bi + bankOffsetB];

}
