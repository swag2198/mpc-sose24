#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAErr(const char* msg);

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// No-operation sorting kernel
//
// This takes in an unordered set, and builds a dummy bucket representation around it
// It does not perform any actual sorting!
//
// This kernel must be launched with a 1,1 configuration (1 grid block, 1 thread).

static __global__ void noSortKernel(const Position* inputPositions, int numInputPositions,
                                    Position* outputPositions, Bucket* outputBuckets)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy contents of input positions into output positions

    for (int i = 0; i < numInputPositions; ++i)
        outputPositions[i] = inputPositions[i];

    // Set up the set of buckets to cover the output positions evenly

    for (int i = 0; i < numBuckets; i++)
    {
        Bucket& bucket = outputBuckets[i];

        bucket.startIndex = numInputPositions * i / numBuckets;
        bucket.numElements = (numInputPositions * (i + 1) / numBuckets) - bucket.startIndex;
    }
}

// !!! missing !!!
// Kernels needed for sortPositionsIntoBuckets(...)

__global__ void reduceKernel (int* featureImage, int* blockSumArray) { 
    /*
    Computes the reduce step of the _inclusive_ parallel scan algorithm.
    Optional blockSumArray Paramters that stores the sum of each block.
    The blockSumArray is used to propagate the sum of each block to the next blocks.
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    int index = x + y * blockDim.x * gridDim.x;

    // We need log(n) iterations to compute the partial sums - we have perfect 2^n block sizes
    for (int stride=1; stride < blockDim.x; stride *= 2) {
        if ((threadIdx.x + 1) % (stride * 2) == 0 && index - stride >= 0) {
            featureImage[index] += featureImage[index - stride];
        }
        __syncthreads();
    }

    // If this is the last thread in the block, store the sum of the block in the blockSumArray
    if (blockSumArray != NULL && threadIdx.x == blockDim.x - 1) {
        blockSumArray[blockIdx.x + blockIdx.y * gridDim.x] = featureImage[index];
    }
}

__global__ void spreadKernel (int* featureImage) {
    // Spreading step for the partial sums accumulated above.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    int index = x + y * blockDim.x * gridDim.x;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if ((threadIdx.x + 1) % (stride * 2) == 0) {
            featureImage[index+stride] += featureImage[index];
        }
        __syncthreads();
    }
}

__global__ void computeHistogramKernel(
    Position* InputPositions, 
    int numInputPositions, 
    int* perBucketIndices,
    int* histogram, 
    int bucketsPerAxis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numInputPositions) {
        int x = (int) (InputPositions[idx].x * bucketsPerAxis);
        int y = (int) (InputPositions[idx].y * bucketsPerAxis);
        perBucketIndices[idx] = atomicAdd(histogram + x + y * bucketsPerAxis, 1); // increment histogram[Bucket_index] by 1
    }
}

__global__ void setBucketKernel(
    Bucket* outputBuckets,
    int* histogram,
    int numBuckets, //256
    int* BucketPrefixes // prefix sum of the histogram
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBuckets) {
        outputBuckets[idx].startIndex = 0;
        if (idx != 0) {
            outputBuckets[idx].startIndex = BucketPrefixes[idx - 1];
        }
        outputBuckets[idx].numElements = histogram[idx];
    }
}

__global__ void sortElementsKernel(
    Position* inputPositions, 
    int numInputPositions, 
    Position* outputPositions, 
    int* perBucketIndices, 
    int* histogram, 
    int bucketsPerAxis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numInputPositions) {
        int x = (int) (inputPositions[idx].x * bucketsPerAxis);
        int y = (int) (inputPositions[idx].y * bucketsPerAxis);
        int bucketIndex = x + y * bucketsPerAxis;
        int outputIndex = perBucketIndices[idx];
        if (bucketIndex != 0) {
            outputIndex += histogram[bucketIndex - 1];
        }
        outputPositions[outputIndex] = inputPositions[idx];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    //
    // 2. Given the histogram, compute where in the output array that each bucket begins, and how
    // large it is
    //    (perform prefix summation over the histogram)
    //
    // 3. Given the start of each bucket within the output array, scatter elements from the input
    //    array into the output array
    //
    // Your new sort implementation should be able to handle at least 10 million entries, and
    //  run in reasonable time (the reference implementations does the job in 200 milliseconds).

    // An integer array to store per bucket the number of elements that should end up in it

    bool debug = false;

    CudaArray<int> cudaHistogram(cudaOutputPositionBuckets.size());
    CudaArray<int> perBucketIndices(cudaInputPositions.size());
    CudaArray<int> cudaBucketPrefixes(cudaOutputPositionBuckets.size());

    // Need to set the histogram to zero since we will be accumulating into it
    cudaMemset(cudaHistogram.cudaArray(), 0, cudaHistogram.size() * sizeof(int));

    // calculate the histogram by incrementing the block sizes
    computeHistogramKernel<<<cudaInputPositions.size() / 256 + 1, 256>>>(
        cudaInputPositions.cudaArray(), cudaInputPositions.size(), 
        perBucketIndices.cudaArray(), cudaHistogram.cudaArray(), BucketsPerAxis);

    if (debug) {
        int cpuHistogram[cudaHistogram.size()];
        cudaHistogram.copyFromCuda(cpuHistogram);
        int sum = 0;
        for (int i = 0; i < cudaHistogram.size(); i++) {
            cout << cpuHistogram[i] << " ";
            sum += cpuHistogram[i];
        }
        cout << endl;
        cout << "sum: " << sum << endl;
        cout << "numInputPositions: " << cudaInputPositions.size() << endl;
    }

    // Copy the histogram into the bucketPrefixes array - we will compute the prefix sum in place
    cudaMemcpy(cudaBucketPrefixes.cudaArray(), cudaHistogram.cudaArray(), cudaHistogram.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    // Compute the prefix sum of the histogram
    reduceKernel<<<1, cudaOutputPositionBuckets.size()>>>(cudaBucketPrefixes.cudaArray(), NULL);
    spreadKernel<<<1, cudaOutputPositionBuckets.size()>>>(cudaBucketPrefixes.cudaArray());

    setBucketKernel<<<1, cudaOutputPositionBuckets.size()>>>(
        cudaOutputPositionBuckets.cudaArray(), cudaHistogram.cudaArray(), cudaOutputPositionBuckets.size(), cudaBucketPrefixes.cudaArray()
    );

    checkCUDAErr("sortPositionsIntoBuckets");

    // Scatter the input positions into the output positions
    sortElementsKernel<<<cudaInputPositions.size() / 256 + 1, 256>>>(
        cudaInputPositions.cudaArray(), cudaInputPositions.size(), 
        cudaOutputPositions.cudaArray(), perBucketIndices.cudaArray(), 
        cudaBucketPrefixes.cudaArray(), BucketsPerAxis);

    //=================  Your code here =====================================
    // !!! missing !!!

    // Instead of sorting, we will now run a dummy kernel that just duplicates the
    //  output positions, and constructs a set of dummy buckets. This is just so that
    //  the test program will not crash when you try to run it.
    //
    // This kernel is run single-threaded because it is throw-away code where performance
    //  does not matter; after all, the purpose of the lab is to replace it with a
    //  proper sort algorithm instead!

    //========== Remove this code when you begin to implement your own sorting algorithm ==========

    // noSortKernel<<<1, 1>>>(cudaInputPositions.cudaArray(), cudaInputPositions.size(),
    //                        cudaOutputPositions.cudaArray(), cudaOutputPositionBuckets.cudaArray());
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);

        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}


void checkCUDAErr(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
