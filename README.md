# CudaExamples
### Examples here are executed on GTX 1050 Ti with Intel's 10th Gen CometLake Processor... 


#### cudeindex ==> Demonstrating the Indexing using Cuda Block and Threads

#### cudaCGgrouping_sumarray ==> Demonstrates adding 16M number using Cooperative Grouping Concept of CUDA ,Alss need to compile it using below nvcc command as atomicAdd_block supported by computer capability 6.1 or above ... 

    nvcc -o arraysum cudaCGgrouping_sumarray.cu  -arch=compute_61
