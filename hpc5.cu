#include <iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void multiply(int* A, int* B, int* C, int N) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

int main() {
    const int N = 4;
    int A[N*N], B[N*N], C[N*N];

    for (int i = 0; i < N*N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    int *dA, *dB, *dC;
    size_t size = N * N * sizeof(int);
    cudaMalloc(&dA, size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 threads(N, N);
    multiply<<<1, threads>>>(dA, dB, dC, N);
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cout << "A:\n"; for (int i = 0; i < N*N; i++) cout << A[i] << ((i+1)%N ? " " : "\n");
    cout << "B:\n"; for (int i = 0; i < N*N; i++) cout << B[i] << ((i+1)%N ? " " : "\n");
    cout << "C:\n"; for (int i = 0; i < N*N; i++) cout << C[i] << ((i+1)%N ? " " : "\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
