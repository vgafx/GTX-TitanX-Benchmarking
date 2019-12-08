#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <getopt.h>


using namespace std;


static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void flush(int *F){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int loop_times = 1572864 / 1024;

    for (int f = 0; f < loop_times; f++){
        int temp = F[(f * 1024)+ i];
        temp++;
        F[(f * 1024)+ i] = temp;
    }

}

__global__ void printTimes(int *P, int *T, int iter){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int w_size = 32;
    //int loop_times = iter / w_size;
    //if(iter == 428160){
        for(int p = 0; p < 4; p++){
            int my_idx = ((p * w_size) + i);
            //printf("Missed Element %d at address %llu latency %d \n", T[my_idx],&P[T[my_idx]] , T[my_idx+32]);
            if(T[my_idx]!=0 ){
            printf("Missed Element %d at address %llu \n", T[my_idx],&P[T[my_idx]]);
        }
        }
    //}
}


__global__ void nugBench(int *P, int *T, int iterations, long long ctrl_base_address){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int loop_times = iterations / 32;
    int *add;
    int temp=0, newtemp=0, one = 1;
    int cmisses=0, hmisses=0;
    unsigned int start, stop;
    __shared__ int accum[32];
    __shared__ int accum2[32];

    __shared__ int misses[2];
    __shared__ int final_accum[2];

    __shared__ int times[2048];
    int sm_offset = 0;



    for (int c = 0; c < loop_times; c++){
        int cur_idx = c *32;
        cur_idx += i;
        asm volatile("membar.cta;");
        add = &P[cur_idx];
        __syncthreads();
        asm volatile("membar.cta;");
        asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(temp) :"l"(add) : "memory");
        asm volatile("add.u32 %0, %1, %2;" : "=r"(temp) : "r"(temp), "r"(one) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(newtemp) : "r"(temp) : "memory");
        asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
        accum[i]+=temp;
        accum2[i]+=newtemp;
        if (stop-start > 335){
            cmisses++;
        }
    }




    asm volatile("membar.cta;");
    temp = 0;
    newtemp = 0;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(temp) :"l"(&P[ctrl_base_address + i]) : "memory");
    asm volatile("add.u32 %0, %1, %2;" : "=r"(temp) : "r"(temp), "r"(one) : "memory");
    asm volatile("mov.u32 %0, %1;" : "=r"(newtemp) : "r"(temp) : "memory");
    accum[i]+=temp;
    accum2[i]+=newtemp;
    __syncthreads();
    asm volatile("membar.cta;");


    for (int h = 0; h < loop_times; h++){
        int cur_idx = h *32;
        cur_idx += i;
        asm volatile("membar.cta;");
        add = &P[cur_idx];
        __syncthreads();
        asm volatile("membar.cta;");
        asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(temp) :"l"(add) : "memory");
        asm volatile("add.u32 %0, %1, %2;" : "=r"(temp) : "r"(temp), "r"(one) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(newtemp) : "r"(temp) : "memory");
        asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
        accum[i]+=temp;
        accum2[i]+=newtemp;
        if (stop-start > 335){
            hmisses++;
            times[i+sm_offset] = (h * 32) + i;
            sm_offset+=32;

        }
    }

    __syncthreads();
    asm volatile("membar.cta;");
    atomicAdd(&misses[0], cmisses);
    atomicAdd(&misses[1], hmisses);
    atomicAdd(&final_accum[0], accum[i]);
    atomicAdd(&final_accum[1], accum2[i]);
    if (i == 0){
        printf("C-Misses: %d\n",misses[0]);
        if (misses[1] > 0) {
            printf("!H-Misses: %d\n",misses[1]); 
        } else {
            printf("H-Misses: %d\n",misses[1]);  
        }

        misses[0] = 0;
        misses[1] = 0;
        printf("T-Ac:%d\n",final_accum[0]);
        printf("NT-Ac:%d\n",final_accum[1]);
        //printf("Address (start) of cline causing eviction: %llu \n", &P[377376]);
        printf("Base Address %llu \n", &P[0]);
        printf("Control Address %llu \n", &P[ctrl_base_address]);
    }

    hmisses = 0;
    cmisses = 0;
    sm_offset=0;

    if(misses[1] <= 64){
        T[i] = times[i];
        T[32+i] = times[32+i]; 
    }


    //T[64+i] = times[64+i];
    //T[96+i] = times[96+i];
    //T[128+i] = times[128+i];
    //T[160+i] = times[160+i];
    //T[192+i] = times[192+i];
    //T[224+i] = times[224+i];
}



void mappingWarpBenchmark(int size, int stride, int iter, long long ctrl_address) {

    int *p = new int[size];
    int *t = new int[size];
    int *f = new int[size];


    int *deviceP = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceP, (size+2) * sizeof(int)));
    if (deviceP == NULL) {
        cout << "could not allocate memoryP!" << endl;
        return;
    }

    int *deviceT = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceT, (size) * sizeof(int)));
    if (deviceT == NULL) {
        cout << "could not allocate memoryP!" << endl;
        return;
    }


    int *deviceF = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceF, (size) * sizeof(int)));
    if (deviceF == NULL) {
        cout << "could not allocate memoryF!" << endl;
        return;
    }

    //Init vector with mem addresses as elements
    for(int i=0; i<size; i++){
        p[i] = (i+stride) % size;
        t[i] = 0;
        f[i] = 11;
    }

    int window_size = 32;
    //wrap around
    p[size]=0;
    p[size+1]=0;

    checkCudaCall(cudaMemcpy(deviceP, p, (size+2)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceT, t, (size)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceF, f, (size)*sizeof(int), cudaMemcpyHostToDevice));

    printf("Flushing L2..\n");
    flush<<<1,1024>>>(deviceF);

    printf("Launching kernel with %d elements, last addr %llu\n",iter, &deviceP[iter-1]);
    printf("Last CLine begins %llu\n", &deviceP[iter-32]);

    nugBench<<<1,32>>>(deviceP,deviceT,iter,ctrl_address);
    //mappingWarpBench<<<1,32>>>(deviceP,deviceT,iter, window_size);
    cudaDeviceSynchronize();
    printTimes<<<1,32>>>(deviceP,deviceT,iter);
    cudaDeviceSynchronize();
    //checkCudaCall(cudaGetLastError());
    delete[] p;
    delete[] t;
    delete[] f;
    checkCudaCall(cudaFree(deviceP));
    checkCudaCall(cudaFree(deviceT));
    checkCudaCall(cudaFree(deviceF));
    cout << "Bench Kernel completed" << endl;
    cout << "     " << endl;
    //printf("Bench Kernel completed\n");
}

int main(int argc, char* argv[]) {

    //L2:3145728 bytes = 786432 ints(32bit)
    int array_size = 2000000;
    int stride = 32; //The stride of mem addresses stored in the vector
    int iterations = 428000;//32; //The number of elements that the 1st run begins with
    int b_stide = 32; //The stride in # of elements for each consecutive benchmark
    int bench_times = 24576; //24576 //How many times to test
    long long in_address=0;

    int ch;
    char *endptr = NULL;
    while ((ch = getopt(argc, argv, "a:i:")) != -1)
    {
        switch(ch) {
        case 'a': in_address = strtol(optarg, &endptr, 10); break;
        case 'i': iterations = strtol(optarg, &endptr, 10); break;
        }
    }


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device name: %s\n", prop.name);
    printf("  L2 size: %d\n", prop.l2CacheSize);
    printf("  Compute Capability(Major.Minor): %d.%d\n", prop.major, prop.minor);
    
    size_t limit=0;
    cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    //printf("Device Printf default limit: %u\n", (unsigned)limit);
    size_t limit1 = 4096000000;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit1);
    size_t limit2=0;
    cudaDeviceGetLimit(&limit2, cudaLimitPrintfFifoSize);
    //printf("Device Printf NEW limit: %u\n", (unsigned)limit2);

    //for (int r = 0; r < bench_times; r++){
    //    if (r != 0){iterations += b_stide;}
        mappingWarpBenchmark(array_size, stride, iterations, in_address);
    //}

    return 0;
}
