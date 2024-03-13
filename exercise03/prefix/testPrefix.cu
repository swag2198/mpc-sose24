// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

__device__ __constant__ float3 gpuClusterCol[2048];

#define THREADS 256
#define LOG_IMG_SIZE 8
#define IMG_SIZE 256
#define WINDOW 6

/* The function measures for every pixel the distance to all
 clusters, and determines the clusterID of the nearest cluster
 center. It then colors the pixel in the cluster's color.

 The cluster centers are given as an array of linear indices into
 the vector image, i.e.    _clusterInfo[0] = (x_0 + y_0 * _w).

 */
__global__ void voronoiKernel(float3* _dst, int _w, int _h, int _nClusters, const int* _clusterInfo)
{
    // get the shared memory
    extern __shared__ int shm[];

    int nIter = _nClusters / THREADS + 1;
    // load cluster data
    for (int i = 0; i < nIter; ++i)
    {
        int pos = i * THREADS + threadIdx.x;
        if (pos < _nClusters)
        {
            shm[pos] = _clusterInfo[pos];
        }
    }

    __syncthreads();

    // compute the position within the image
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y;

    int pos = x + y * _w;

    // determine which is the closest cluster
    float minDist = 1000000.;
    int minIdx = 0;
    for (int i = 0; i < _nClusters; ++i)
    {

        float yy = shm[i] >> LOG_IMG_SIZE;
        float xx = shm[i] % IMG_SIZE;

        float dist = (x - xx) * (x - xx) + (y - yy) * (y - yy);
        if (dist < minDist)
        {
            minDist = dist;
            minIdx = i;
        }
    }

    _dst[pos].x = gpuClusterCol[minIdx].x;
    _dst[pos].y = gpuClusterCol[minIdx].y;
    _dst[pos].z = gpuClusterCol[minIdx].z;

    // mark the center of each cluster
    if (minDist <= 2.)
    {
        _dst[pos].x = 255;
        _dst[pos].y = 0.;
        _dst[pos].z = 0.;
    }
}

__device__ float luminance(const float4& _col)
{
    return 0.299 * _col.x + 0.587 * _col.y + 0.114 * _col.z;
}

/** stores a 1 in _dst if the pixel's luminance is a maximum in the
WINDOW x WINDOW neighborhood
 */
__global__ void featureKernel(int* _dst, cudaTextureObject_t texImg, int _w, int _h)
{
    // compute the position within the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    float lum = luminance(tex2D<float4>(texImg, x, y));

    bool maximum = false;

    if (lum > 20)
    {
        maximum = true;
        for (int v = y - WINDOW; v < y + WINDOW; ++v)
        {
            for (int u = x - WINDOW; u < x + WINDOW; ++u)
            {

                if (lum < luminance(tex2D<float4>(texImg, u, v)))
                {
                    maximum = false;
                }
            }
        }
    }

    if (maximum)
    {
        _dst[x + y * _w] = 1;
    }
    else
    {
        _dst[x + y * _w] = 0;
    }
}

// !!! missing !!!
// Kernels for Prefix Sum calculation (compaction, spreading, possibly shifting)
// and for generating the gpuFeatureList from the prefix sum.

__global__ void reduceKernel (int* featureImage, int* blockSumArray) { 
    /*
    Computes the reduce step of the _inclusive_ parallel scan algorithm.
    Optional blockSumArray Paramters that stores the sum of each block.
    The blockSumArray is used to propagate the sum of each block to the next blocks.
    */
    int x = threadIdx.x;
    int y = blockIdx.x;

    int index = x + y * blockDim.x;

    // if (threadIdx.x == blockDim.x - 1 && blockIdx.x == 73) {
    //     printf("before stride 1");
    //     for (int i = 0; i < blockDim.x; i++) {
    //         printf("%d ", featureImage[i + blockDim.x * blockIdx.x]);
    //     }
    //     printf("\n");
    // }

    // We need log(n) iterations to compute the partial sums - we have perfect 2^n block sizes
    for (int stride=1; stride < blockDim.x; stride *= 2) {
        if ((threadIdx.x + 1) % (stride * 2) == 0 && index - stride >= 0) {
            featureImage[index] += featureImage[index - stride];
        }
        __syncthreads();

        // if (threadIdx.x == blockDim.x - 1 && blockIdx.x == 73) {
        //     printf("after stride %d\n", stride);
        //     for (int i = 0; i < blockDim.x; i++) {
        //         printf("%d ", featureImage[i + blockDim.x * blockIdx.x]);
        //     }
        //     printf("\n");
        // }
    }

    // If this is the last thread in the block, store the sum of the block in the blockSumArray
    if ((blockSumArray != NULL) && (threadIdx.x == blockDim.x - 1)) {
        blockSumArray[blockIdx.x] = featureImage[index];
    }
}

__global__ void spreadKernel (int* featureImage) {
    // Spreading step for the partial sums accumulated above.
    int x = threadIdx.x;
    int y = blockIdx.x;

    int index = x + y * blockDim.x;

    if (threadIdx.x == 0 && blockIdx.x == 73) {
        printf("before stride 1\n");
        for (int i = 0; i < blockDim.x; i++) {
            printf("%d ", featureImage[i + blockDim.x * blockIdx.x]);
        }
        printf("\n");
    }


    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if ((threadIdx.x + 1) % (stride * 2) == 0) {
            if(index + stride < blockDim.x*blockIdx.x + 256)
                featureImage[index+stride] += featureImage[index];
        }
        __syncthreads();

        if (threadIdx.x == 0 && blockIdx.x == 73) {
            printf("after stride %d\n", stride);
            for (int i = 0; i < blockDim.x; i++) {
                printf("%d ", featureImage[i + blockDim.x * blockIdx.x]);
            }
            printf("\n");
        }
    }
}

__global__ void computeFeatureList (int* featureImage, int* blockSumArray, int* featureList) {
    /*
    Computes the final feature list from the prefix sum of the featureImage.
    Uses the blockSumArray to offset the indices of the features by the sum of the previous blocks.
    */
    int x = threadIdx.x;
    int y = blockIdx.x;

    int index = x + y * blockDim.x;

    // We can use a shared variable to store the sum of the previous blocks - it is read by all threads in the block.
    // __shared__ int prevBlocksSum;
    // prevBlocksSum = 0;
    
    // if (threadIdx.x == 0 && blockIdx.y > 0) {
    //     // The first thread is responsible for reading the sum of the previous blocks from the blockSumArray
    //     prevBlocksSum = blockSumArray[blockIdx.x - 1 + blockIdx.y * gridDim.x];
    // }
    // __syncthreads();
    int prevBlocksSum;
    if (blockIdx.x > 0){
        prevBlocksSum = blockSumArray[blockIdx.x - 1];
    } else {
        prevBlocksSum = 0;
    }

    __syncthreads();
    // If the current pixel is a feature, store its index in the featureList
    if (((threadIdx.x == 0) && (featureImage[index] == 1)) || ((threadIdx.x > 0) && (featureImage[index] == (featureImage[index-1] + 1)))) {
        // The index of the feature is the index of the current pixel (in current block) + the sum of the previous blocks
        featureList[featureImage[index] + prevBlocksSum - 1] = index;
    }
}

/* This program detects the local maxima in an image, writes their
location into a vector and then computes the Voronoi diagram of the
image given the detected local maxima as cluster centers.

A Voronoi diagram simply colors every pixel with the color of the
nearest cluster center. */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;
    if (argc < 4)
    {
        printf("usage: testPrefix <inImg> <outImg> <mode>\n");
        exit(1);
    }
    string inName(argv[acount++]);
    string outName(argv[acount++]);
    int mode = atoi(argv[acount++]);

    // Load the input image
    float* cpuImage;
    int w, h;
    readPPM(inName.c_str(), w, h, &cpuImage);
    int nPix = w * h;

    // Allocate GPU memory
    int* gpuFeatureImg; // Contains 1 for a feature, 0 else
    // Can be used to do the reduction step of prefix sum calculation in place
    int* gpuPrefixSumShifted; // Output buffer containing the prefix sum
    // Shifted by 1 since it contains 0 as first element by definition
    int* gpuFeatureList; // List of pixel indices where features can be found.
    float3* gpuVoronoiImg; // Final rgb output image
    cudaMalloc((void**)&gpuFeatureImg, (nPix) * sizeof(int));

    cudaMalloc((void**)&gpuPrefixSumShifted, (nPix + 1) * sizeof(int));
    cudaMalloc((void**)&gpuFeatureList, 10000 * sizeof(int));

    cudaMalloc((void**)&gpuVoronoiImg, nPix * 3 * sizeof(float));

    // color map for the cluster
    float clusterCol[2048 * 3];
    float* ci = clusterCol;
    for (int i = 0; i < 2048; ++i, ci += 3)
    {
        ci[0] = 32 * i % 256;
        ci[1] = (10 * i + 128) % 256;
        ci[2] = (40 * i + 255) % 256;
    }

    cudaMemcpyToSymbol(gpuClusterCol, clusterCol, 2048 * 3 * sizeof(float));  // what does it do?

    cudaArray* gpuTex;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&gpuTex, &floatTex, w, h);

    // pad to float4 for faster access
    float* img4 = new float[w * h * 4];

    for (int i = 0; i < w * h; ++i)
    {
        img4[4 * i] = cpuImage[3 * i];
        img4[4 * i + 1] = cpuImage[3 * i + 1];
        img4[4 * i + 2] = cpuImage[3 * i + 2];
        img4[4 * i + 3] = 0.;
    }

    // upload to array
    cudaMemcpy2DToArray(gpuTex, 0, 0, img4, w * 4 * sizeof(float), w * 4 * sizeof(float), h,
                        cudaMemcpyHostToDevice);

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpuTex;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    cout << "setup texture" << endl;
    cout.flush();

    // calculate the block dimensions
    dim3 threadBlock(THREADS);
    dim3 blockGrid(w / THREADS, h, 1);

    printf("blockDim: %d  %d \n", threadBlock.x, threadBlock.y);
    printf("gridDim: %d  %d \n", blockGrid.x, blockGrid.y);

    featureKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg, tex, w, h);

    // variable to store the number of detected features = the number of clusters
    int nFeatures;

    if (mode == 0)
    {
        ////////////////////////////////////////////////////////////
        // CPU compaction:
        ////////////////////////////////////////////////////////////

        // download result

        cudaMemcpy(cpuImage, gpuFeatureImg, nPix * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<int> features;

        float* ii = cpuImage;
        for (int i = 0; i < nPix; ++i, ++ii)
        {
            if (*ii > 0)
            {
                features.push_back(i);
            }
        }

        cout << "nFeatures: " << features.size() << endl;

        nFeatures = features.size();
        // upload feature vector

        cudaMemcpy(gpuFeatureList, &(features[0]), nFeatures * sizeof(int), cudaMemcpyHostToDevice);
    }
    else
    {
        ////////////////////////////////////////////////////////////
        // GPU compaction:
        ////////////////////////////////////////////////////////////

        // An array to store the sum of each block
        int* blockSumArray;

        bool DEBUG = false;

        cudaMalloc((void **)&blockSumArray, sizeof(int) * blockGrid.x * blockGrid.y);

        if (DEBUG) {
            int sz = 256;
            int testPointer[sz];
            // int testPointer[] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < sz; i++) {
                testPointer[i] = 1;
            }

            dim3 threadBlock(sz);
            dim3 blockGrid(1, 1, 1);
            cudaMemcpy(gpuFeatureImg, testPointer, sizeof(int) * sz, cudaMemcpyHostToDevice);
            reduceKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg, NULL);
            spreadKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg);
            cudaMemcpy(testPointer, gpuFeatureImg, sizeof(int) * sz, cudaMemcpyDeviceToHost);

            for (int i = 0; i < sz; i++) {
                cout << testPointer[i] << ' ';
            }
            exit(1);
        } else {

            int fi[256];
            cudaMemcpy(fi, gpuFeatureImg + (73 * 256), sizeof(int)* 256, cudaMemcpyDeviceToHost);
            cout<< "Before reduce:" <<endl;

            for(int i = 0; i < 256; i++) {
                cout << fi[i] << ' ';
            }
            cout<< endl;

            // Compute the reduction sum linewise, copy the blockwise sums to the blockSumArray
            reduceKernel<<<256, threadBlock>>>(gpuFeatureImg, blockSumArray);
            cudaDeviceSynchronize();
            cudaMemcpy(fi, gpuFeatureImg + (73 * 256), sizeof(int)* 256, cudaMemcpyDeviceToHost);
            cout<< "After reduce:" <<endl;
            for(int i = 0; i < 256; i++) {
                cout << fi[i] << ' ';
            }
            cout<< endl;

            spreadKernel<<<256, threadBlock>>>(gpuFeatureImg);

            cudaMemcpy(fi, gpuFeatureImg + (73 * 256), sizeof(int)* 256, cudaMemcpyDeviceToHost);
            for(int i = 0; i < 256; i++) {
                cout << fi[i] << ' ';
            }
            cout<< endl;

            // Compute the prefix sum of the blockSumArray - 
            reduceKernel<<<1, threadBlock>>>(blockSumArray, NULL);
            spreadKernel<<<1, threadBlock>>>(blockSumArray);

            int bs[256];
            cudaMemcpy(bs, blockSumArray, sizeof(int)* 256, cudaMemcpyDeviceToHost);
            for(int i = 0; i < 256; i++) {
                cout << bs[i] << ' ';
            }
            cout<< endl << "FEATLIST" << endl;

            // Compute the final feature list - the indices of the features are offset by the sum of the previous blocks
            computeFeatureList<<<256, threadBlock>>>(gpuFeatureImg, blockSumArray, gpuFeatureList);
            cudaMemcpy(&nFeatures, blockSumArray + blockGrid.x * blockGrid.y - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cout << "N Features " << nFeatures << endl;
            int fl[nFeatures];
            cudaMemcpy(fl, gpuFeatureList, sizeof(int)* nFeatures, cudaMemcpyDeviceToHost);
            for(int i = 0; i < nFeatures; i++) {
                cout << fl[i] << ' ';
            }
            cout<< endl;

        }

        cudaFree(blockSumArray);
    }

    // now compute the Voronoi Diagram around the detected features.
    voronoiKernel<<<blockGrid, threadBlock, nFeatures * sizeof(int)>>>(gpuVoronoiImg, w, h,
                                                                       nFeatures, gpuFeatureList);

    // download final voronoi image.
    cudaMemcpy(cpuImage, gpuVoronoiImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // Write to disk
    writePPM(outName.c_str(), w, h, (float*)cpuImage);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(gpuTex);
    cudaFree(gpuFeatureList);
    cudaFree(gpuFeatureImg);
    cudaFree(gpuPrefixSumShifted);
    cudaFree(gpuVoronoiImg);

    delete[] cpuImage;
    delete[] img4;

    checkCUDAError("end of program");

    printf("done\n");
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
