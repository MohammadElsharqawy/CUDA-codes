#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void vectorAdd(int* A, int* B, int* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // your code goes here

    int a[] = { 1,2,3 };
    int b[] = { 4,5,6 };
    int c[sizeof(a) / sizeof(int)] = { 0 };

    // create parameters into gpu
    int* cudaA = 0;
    int* cudaB = 0;
    int* cudaC = 0;

    // allocate memory into gpu
    cudaMalloc(&cudaA, sizeof(a));
    cudaMalloc(&cudaB, sizeof(b));
    cudaMalloc(&cudaC, sizeof(c));

    // copy the vectors into gpu
    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

    vectorAdd<<<1, sizeof(a) / sizeof(int)>>> (cudaA, cudaB, cudaC);

    cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

    for (int i : c)
        std::cout << i << std::endl;

    // Free allocated memory
    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    return 0;
}
