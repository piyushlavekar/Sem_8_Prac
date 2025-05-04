#include <iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void add(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 4;
    int A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    int *dA, *dB, *dC;
    size_t size = N * sizeof(int);
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    add<<<1, N>>>(dA, dB, dC, N);
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cout << "A: "; for (int i : A) cout << i << " ";
    cout << "\nB: "; for (int i : B) cout << i << " ";
    cout << "\nC: "; for (int i : C) cout << i << " ";

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}


// A = (int*)malloc(size);
// B = (int*)malloc(size);
// C = (int*)malloc(size);
