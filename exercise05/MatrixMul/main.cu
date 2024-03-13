#include "Tools.h"

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// #define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
// #define RUN_CPU // Runs CPU code for reference (slow!!!)
#define N 1024 // Must be a multiple of THREADS_PER_BLOCK default=1024
#define THREADS_PER_BLOCK 32 // per axis -> block has this value squared threads.
void multiplyMatrix(float* result, const float* a, const float* b, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            result[i * n + j] = 0.0f;
            for (unsigned int k = 0; k < n; k++)
            {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void dumpMatrix(const float* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
        }
        cout << endl;
    }
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

__global__ void multiplyMatrixGpu1(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a trivial GPU square matrix multiplication.
    // Use one thread per output element.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float res_ij = 0;
    if ((i < n) && (j < n))
    {        
        for(int k=0; k<n; k++)
            res_ij += (a[i*n+k] * b[k*n+j]);
    }
    result[i*n + j] = res_ij;
}

__global__ void multiplyMatrixGpu2(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a more sophisticated GPU square matrix multiplication.
    // Compute square submatrices per block. Load the common input
    // data of all threads of a block into shared memory cooperatively.
    __shared__ float a_shared[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ float b_shared[THREADS_PER_BLOCK][THREADS_PER_BLOCK];

    int bi = blockIdx.y;
    int bj = blockIdx.x;
    int ti = threadIdx.y;
    int tj = threadIdx.x;

    // int start_i = bi * blockDim.y;
    // int start_j = bj * blockDim.x;

    // this thread calculates output at [i, j]
    int i = bi*blockDim.y + ti;
    int j = bj*blockDim.x + tj;

    float res_ij = 0;
    // copy one block at a time and compute partial results
    for(int k=0; k < n/THREADS_PER_BLOCK; k++)
    {
        // fill block-k
        // a[i][0--n-1]
        // b[0---n-1][j]

        // a_shared[ti][tj] = a[i][k*T + tj]
        // b_shared[ti][tj] = b[k*T + ti][j];
        // a[i][j] == a[i*n + j]
        a_shared[ti][tj] = a[i*n + k*THREADS_PER_BLOCK + tj];
        b_shared[ti][tj] = b[(k*THREADS_PER_BLOCK + ti)*n + j];
        __syncthreads();

        for(int l=0; l<THREADS_PER_BLOCK; l++)
            res_ij += a_shared[ti][l] * b_shared[l][tj];

        __syncthreads();
    }

    result[i*n + j] = res_ij;

}

int verify_result(float *res1, float *res2, int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(abs(res1[n*i + j] - res2[n*i + j]) > 1e-4)
            {
                cout << i << j <<endl;
                cout<<"res1 val = "<<setw(10) << setprecision(10)<<res1[n*i + j]<<endl;
                cout<<"res2 val = "<<setw(10) << setprecision(10)<<res2[n*i + j]<<endl;
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char** argv)
{
    __int64_t startTime;
    __int64_t endTime;

    // Allocate all memory
    float* hM1 = new float[N * N];
    float* hM2 = new float[N * N];
    float* hMR = new float[N * N];
    float* hMR_gpu1 = new float[N * N];
    float* hMR_gpu2 = new float[N * N];
    float* gM1;
    cudaMalloc(&gM1, sizeof(float) * N * N);
    float* gM2;
    cudaMalloc(&gM2, sizeof(float) * N * N);
    float* gMR;
    cudaMalloc(&gMR, sizeof(float) * N * N);

    // Initialize matrices and upload to CUDA
    for (unsigned int n = 0; n < N * N; n++)
    {
        hM1[n] = randF(-1.0, 1.0);
        hM2[n] = randF(-1.0, 1.0);
        //         hM1[n] = 1.0;
        // hM2[n] = 1.0;
    }
    cudaMemcpy(gM1, hM1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gM2, hM2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
#ifdef VERBOSE
    cout << "Input Matrices:" << endl;
    dumpMatrix(hM1, N);
    cout << endl;
    dumpMatrix(hM2, N);
    cout << endl << endl;
#endif

#ifdef RUN_CPU
    // Calculations on CPU
    startTime = continuousTimeNs();
    multiplyMatrix(hMR, hM1, hM2, N);
    endTime = continuousTimeNs();
#ifdef VERBOSE
    cout << "CPU:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout <<"Matrix size: "<< N << endl;
    cout << "CPU time: " << (endTime - startTime) << "ns = " << (endTime - startTime)*1e-6 << "ms" <<endl;
#endif

    // Calculations on GPU
    int blocksPerGridX =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    int blocksPerGridY =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    startTime = continuousTimeNs();
    multiplyMatrixGpu1<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    // cout <<"#blocks in grid: "<<blocksPerGridX<<endl;
    cudaMemcpy(hMR_gpu1, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU simple:" << endl;
    dumpMatrix(hMR_gpu1, N);
    cout << endl;
#endif
    // if(verify_result(hMR_gpu1, hMR, N))
    cout << "GPU simple time: " << (endTime - startTime) << "ns = " << (endTime - startTime)*1e-6 << "ms" << endl;
    // cout << "GPU simple time: " << (endTime - startTime)*1e-6 << "ms" << endl;
    startTime = continuousTimeNs();
    multiplyMatrixGpu2<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR_gpu2, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU advanced:" << endl;
    dumpMatrix(hMR_gpu2, N);
    cout << endl;
#endif
    // if(verify_result(hMR_gpu2, hMR, N) && verify_result(hMR_gpu1, hMR_gpu2, N))
    cout << "GPU advanced time: " << (endTime - startTime) << "ns = " << (endTime - startTime)*1e-6 << "ms" << endl;
    // cout << "GPU advanced time: " << (endTime - startTime)*1e-6 << "ms" << endl;

    // Free all memory
    cudaFree(gM1);
    cudaFree(gM2);
    cudaFree(gMR);
    delete[] hM1;
    delete[] hM2;
    delete[] hMR;

    checkCUDAError("end of program");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
