// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime_api.h>

#define cudaCheckError(code) \
{                                                                         \
     if((code) != cudaSuccess) {                                          \
        fprintf(stderr, "Cuda failure %s:%d: '%s' \n",__FILE__,__LINE__ , \
            cudaGetErrorString(code));                                    \
     }                                                                    \
}                                                                                       

__global__ void kernel_1d(){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("1D indexing demonstration");
    printf("block %d,blockdim %x,thread %d,index %d\n",blockIdx.x,blockDim.x,threadIdx.x,index);   
    //printf("block %d,thread %d,index %d\n",blockIdx.x,threadIdx.x,index);   
}
__global__ void kernel_2d(){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("2D indexing demonstration");
    printf("blockidx.x %d blockidx.y %d\n ",blockIdx.x,blockIdx.y);
    printf("blockdim.x %d blockdim.y %d\n ",blockDim.x,blockDim.y);
    printf("block.x %d,blockdim.x %x,thread.x %d,x %d\n",blockIdx.x,blockDim.x,threadIdx.x,x);  
    printf("block.y %d,blockdim.y %x,thread.y %d,y %d\n",blockIdx.y,blockDim.y,threadIdx.y,y);   
    //printf("block %d,thread %d,index %d\n",blockIdx.x,threadIdx.x,index);   
}

int main(){

    kernel_1d<<<4,8>>>(); 
    kernel_2d<<<(2,3),(3,4)>>>();
    cudaCheckError(cudaDeviceSynchronize());
}
