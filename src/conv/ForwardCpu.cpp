// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "ForwardCpu.h"
#include <ppl.h>
#include "util\SIMDHelper.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

ForwardCpu::ForwardCpu(EasyCL *cl, LayerDimensions dim) :
    Forward(cl, dim)
{
    DeviceInfo info;
    info.populate(cl->platform_id, cl->device);

    if (info.deviceType != CL_DEVICE_TYPE_CPU)
    {
        //throw runtime_error("This kernel is for CPU only");
    }

    if (dim.filterSize == dim.inputSize && dim.outputSize == 1 && dim.padZeros == false && dim.skip == false)
    {
        kernelUsed = 1;
    }
    else if (dim.inputSize == 3 && dim.filterSize == 3 && dim.padZeros == true && dim.skip == 0 && dim.biased == true)
    {
        kernelUsed = 3;
    }
    else if (dim.filterSize == 3 && dim.padZeros == true && dim.skip == 0 && dim.biased == true)
    {
        kernelUsed = 2;
    }
}
VIRTUAL void ForwardCpu::forward(int batchSize, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper)
{
    inputDataWrapper->copyToHost();
    weightsWrapper->copyToHost();
    float *bias = 0;
    if (dim.biased)
    {
        biasWrapper->copyToHost();
        bias = (float *)biasWrapper->getHostArray();
    }

    float * input = (float*)inputDataWrapper->getHostArray();
    float * weights = (float*)weightsWrapper->getHostArray();
    float * output = (float*)outputWrapper->getHostArray();

    switch (kernelUsed)
    {
        case 1: // Fully connected layer
            forwardFC(batchSize, input, weights, bias, output);
            break;
        case 2: // 3x3, padding, no skip, with bias
            forward3x3(batchSize, input, weights, bias, output);
            break;
        case 3:
            forward3x3_3x3(batchSize, input, weights, bias, output);
            break;
        default:
            forward(batchSize, input, weights, bias, output);
            break;
    }

    outputWrapper->copyToDevice();
}

VIRTUAL void ForwardCpu::forwardFC(int batchSize, float *inputData, float *weights, float *bias, float* output)
{
    const int numFilters = dim.numFilters;
    const int size = batchSize * dim.numFilters;
    const int totalNumber = dim.inputPlanes * dim.inputSizeSquared;

    Concurrency::parallel_for(0, size, [&](int index)
    {
        const int n = index / numFilters;
        const int filter = index % numFilters;

        __m256 sum = _mm256_set1_ps(0);

        const int inputIndexStart = (n * dim.inputPlanes) * dim.inputSize;
        const int weightIndexStart = (filter * dim.inputPlanes) * dim.filterSize;

        float* startInput = &inputData[inputIndexStart];
        float* startWeight = &weights[weightIndexStart];

        const int stepSize = 8;
        const int roundedDown = ROUND_DOWN(totalNumber, stepSize);
        int i = 0;
        for (; i < roundedDown; i += stepSize, startInput += stepSize, startWeight += stepSize)
        {
            __m256 result = SIMD_256::MultiplyFloats(SIMD_256::LoadFloat8(startInput[0]), SIMD_256::LoadFloat8(startWeight[0]));

            sum = SIMD_256::AddFloats(sum, result);
        }

        float result = SIMD_256::ToFloat(sum);
        for (; i < totalNumber; i++, startInput++, startWeight++)
        {
            result += startInput[0] * startWeight[0];
        }

        if (dim.biased)
            result += bias[filter];

        output[index] = result;
    });
}

VIRTUAL void ForwardCpu::forward(int batchSize, float *inputData, float *weights, float *bias, float* output)
{
    const int numFilters = dim.numFilters;
    const int size = batchSize * dim.numFilters;
    const int inputMinOne = dim.inputSize - 1;

    const int tempCalc = (dim.skip + 1) + (dim.padZeros ? 0 : dim.halfFilterSize);

    const int stepSize = 1 + dim.skip;

    Concurrency::parallel_for(0, size, [&](int i)
    {
        const int n = i / numFilters;
        const int filter = i % numFilters;
        const float startSum = dim.biased ? bias[filter] : 0;
        float sum = 0;

        // Reuse these to save time creating and deleting them;
        int tempInRow = 0;
        int tempInCol = 0;

        const int inputOffset = n * dim.inputPlanes;
        const int weightOffset = filter * dim.inputPlanes;

        int outputTemp = (n * dim.numFilters + filter) * dim.outputSize;

        int inRow = 0;

        for (int outRow = 0; outRow < dim.outputSize; outRow += stepSize)
        {
            tempInRow = tempCalc != 1 ? outRow * (dim.skip + 1) + (dim.padZeros ? 0 : dim.halfFilterSize) : outRow;

            int maxUp = (tempInRow - dim.halfFilterSize) < 0 ? tempInRow : dim.halfFilterSize;
            int maxDown = (tempInRow + dim.halfFilterSize) > inputMinOne ? (inputMinOne - tempInRow) : dim.halfFilterSize;

            for (int outCol = 0; outCol < dim.outputSize; outCol += stepSize)
            {
                tempInCol = tempCalc != 1 ? outCol * (dim.skip + 1) + (dim.padZeros ? 0 : dim.halfFilterSize) : outCol;

                int maxLeft = (tempInCol - dim.halfFilterSize) < 0 ? tempInCol : dim.halfFilterSize;
                int maxRight = (tempInCol + dim.halfFilterSize) > inputMinOne ? (inputMinOne - tempInCol) : dim.halfFilterSize;
                int temp = maxLeft + maxRight + 1;

                sum = startSum;
                __m128 vSum = _mm_set_ps1(0);
                for (int u = -maxUp; u <= maxDown; u++)
                {
                    inRow = tempInRow + u;

                    int filterRow = u + dim.halfFilterSize;
                    #define inputIndex(plane) (((inputOffset + plane) * dim.inputSize + inRow) * dim.inputSize + inCol)
                    #define weightIndex(plane) (((weightOffset + plane) * dim.filterSize + filterRow) * dim.filterSize + filterCol)
                    switch (temp)
                    {
                        case 2:
                        {
                            int inCol = tempInCol - maxLeft;
                            int filterCol = -maxLeft + dim.halfFilterSize;

                            int startInputIndex = ((inputOffset * dim.inputSize + inRow) * dim.inputSize + inCol);
                            const int inputStepSize = dim.inputSizeSquared;

                            int startWeightIndex = ((weightOffset * dim.filterSize + filterRow) * dim.filterSize + filterCol);
                            const int weightStepSize = dim.filterSizeSquared;

                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInputIndex += inputStepSize, startWeightIndex += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat2(inputData[startInputIndex]), SIMD_128::LoadFloat2(weights[startWeightIndex])), vSum);
                            }
                        }
                        break;
                        case 3:
                        {
                            int inCol = tempInCol - maxLeft;
                            int filterCol = -maxLeft + dim.halfFilterSize;

                            int startInputIndex = ((inputOffset * dim.inputSize + inRow) * dim.inputSize + inCol);
                            const int inputStepSize = dim.inputSizeSquared;

                            int startWeightIndex = ((weightOffset * dim.filterSize + filterRow) * dim.filterSize + filterCol);
                            const int weightStepSize = dim.filterSizeSquared;

                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInputIndex += inputStepSize, startWeightIndex += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat3(inputData[startInputIndex]), SIMD_128::LoadFloat3(weights[startWeightIndex])), vSum);
                            }
                        }
                        break;
                        default:
                        {
                            for (int v = -maxLeft; v <= maxRight; v++)
                            {
                                int inCol = tempInCol + v;
                                int filterCol = v + dim.halfFilterSize;

                                for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++)
                                {
                                    sum += inputData[inputIndex(inPlane)] * weights[weightIndex(inPlane)];
                                }
                            }
                        }
                        break;
                    }
                }

                sum += SIMD_128::ToFloat(vSum);

                int outputIndex = (outputTemp
                    + outRow)
                    * dim.outputSize + outCol;

                output[outputIndex] = sum;
            }
        }
    });
}

VIRTUAL void ForwardCpu::forward3x3_3x3(int batchSize, float *inputData, float *weights, float *bias, float* output)
{
    const int numFilters = dim.numFilters;
    const int size = batchSize * dim.numFilters;
    const int inputMinOne = dim.inputSize - 1;

    const int inputStepSize = dim.inputSizeSquared;
    const int weightStepSize = 3 * 3;

    Concurrency::parallel_for(0, size, [&](int i)
    {
        const int n = i / numFilters;
        const int filter = i % numFilters;
        const float startSum = bias[filter];
        float sum = 0;

        const int inputOffset = (n * dim.inputPlanes) * dim.inputSize;
        const int weightOffset = (filter * dim.inputPlanes) * 3;

        const int outputTemp = (n * dim.numFilters + filter) * dim.outputSize;

        for (int outRow = 0; outRow < dim.outputSize; outRow++)
        {
            const int maxUp = (outRow - 1) < 0 ? outRow : 1;
            const int maxDown = (outRow + 1) > inputMinOne ? (inputMinOne - outRow) : 1;
            const int totalUpDown = maxUp + maxDown + 1;

            for (int outCol = 0; outCol < dim.outputSize; outCol++)
            {
                const int maxLeft = (outCol - 1) < 0 ? outCol : 1;
                const int maxRight = (outCol + 1) > inputMinOne ? (inputMinOne - outCol) : 1;
                const int totalLeftRight = maxLeft + maxRight + 1;

                sum = startSum;
                __m128 vSum = _mm_setzero_ps();
                __m256 vSum256 = _mm256_setzero_ps();

                if (totalLeftRight == 3 && totalUpDown == 3)
                {
                    const int inRow = outRow - 1;
                    const int inCol = outCol - 1;

                    const int startWeightIndex = weightOffset * 3;
                    int startLocation = ((inputOffset + inRow) * dim.inputSize + inCol);

                    float* startInput = &inputData[startLocation];
                    float* startWeight = &weights[startWeightIndex];

                    const int endIndex = startLocation + (inputStepSize * dim.inputPlanes);
                    const int roundedEnd = ROUND_DOWN(endIndex, 8);
                    for (; startLocation < roundedEnd; startLocation += 8, startInput += 8, startWeight += 8)
                    {
                        __m256 input = _mm256_load_ps(&startInput[0]);
                        __m256 weightsData = _mm256_load_ps(&startWeight[0]);

                        vSum256 = SIMD_256::AddFloats(SIMD_256::MultiplyFloats(weightsData, input), vSum256);
                    }

                    for (; startLocation < endIndex; startLocation++, startInput++, startWeight++)
                    {
                        sum += startWeight[0] * startInput[0];
                    }
                }
                else if (totalLeftRight == 2 && totalUpDown == 2)
                {
                    const int inRow = outRow - maxUp;
                    const int filterRow = -maxUp + 1;

                    const int inCol = outCol - maxLeft;
                    const int filterCol = -maxLeft + 1;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat2x2(startInput[0], startInput[dim.inputSize]), SIMD_128::LoadFloat2x2(startWeight[0], startWeight[3])), vSum);
                    }
                }
                else if (totalLeftRight == 3 && totalUpDown == 2)
                {
                    const int inRow = outRow - maxUp;
                    const int filterRow = -maxUp + 1;

                    const int inCol = outCol - 1;
                    const int filterCol = 0;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        __m128 w1 = _mm_load_ps(&startWeight[0]);
                        __m128 w2 = SIMD_128::LoadFloat2(startWeight[4]);

                        __m128 i1 = _mm_load_ps(&startInput[0]);
                        __m128 i2 = SIMD_128::LoadFloat2(startInput[4]);

                        vSum256 = SIMD_256::AddFloats(SIMD_256::MultiplyFloats(SIMD_256::From128(w1, w2), SIMD_256::From128(i1, i2)), vSum256);
                    }
                }
                else if (totalLeftRight == 2 && totalUpDown == 3)
                {
                    const int inRow = outRow - 1;
                    const int filterRow = 0;

                    const int inCol = outCol - maxLeft;
                    const int filterCol = -maxLeft + 1;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        __m128 i1 = SIMD_128::LoadFloat2x2(startInput[0], startInput[dim.inputSize]);
                        __m128 i2 = SIMD_128::LoadFloat2(startInput[dim.inputSize + dim.inputSize]);

                        __m128 w1 = SIMD_128::LoadFloat2x2(startWeight[0], startWeight[3]);
                        __m128 w2 = SIMD_128::LoadFloat2(startWeight[6]);
                        
                        vSum = SIMD_128::AddFloats(SIMD_128::AddFloats(SIMD_128::MultiplyFloats(i1, w1), SIMD_128::MultiplyFloats(i2, w2)), vSum);
                    }
                }
                else
                {
                    for (int u = -maxUp; u <= maxDown; u++)
                    {
                        const int inRow = outRow + u;
                        const int filterRow = u + 1;

                        const int inCol = outCol - maxLeft;
                        const int filterCol = -maxLeft + 1;

                        const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                        const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                        float* startInput = &inputData[startInputIndex];
                        float* startWeight = &weights[startWeightIndex];

                        if (totalLeftRight == 3)
                        {
                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat3(startInput[0]), SIMD_128::LoadFloat3(startWeight[0])), vSum);
                            }
                        }
                        else
                        {
                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat2(startInput[0]), SIMD_128::LoadFloat2(startWeight[0])), vSum);
                            }
                        }
                    }
                }

                vSum = SIMD_128::AddFloats(_mm256_extractf128_ps(vSum256, 0), vSum);
                vSum = SIMD_128::AddFloats(_mm256_extractf128_ps(vSum256, 1), vSum);
                sum += SIMD_128::ToFloat(vSum);

                int outputIndex = (outputTemp
                    + outRow)
                    * dim.outputSize + outCol;

                output[outputIndex] = sum;
            }
        }
    });
}

VIRTUAL void ForwardCpu::forward3x3(int batchSize, float *inputData, float *weights, float *bias, float* output)
{
    const int numFilters = dim.numFilters;
    const int size = batchSize * dim.numFilters;
    const int inputMinOne = dim.inputSize - 1;

    const int inputStepSize = dim.inputSizeSquared;
    const int weightStepSize = 3 * 3;

    Concurrency::parallel_for(0, size, [&](int i)
    {
        const int n = i / numFilters;
        const int filter = i % numFilters;
        const float startSum = bias[filter];
        float sum = 0;

        const int inputOffset = (n * dim.inputPlanes) * dim.inputSize;
        const int weightOffset = (filter * dim.inputPlanes) * 3;

        const int outputTemp = (n * dim.numFilters + filter) * dim.outputSize;

        for (int outRow = 0; outRow < dim.outputSize; outRow++)
        {
            const int maxUp = (outRow - 1) < 0 ? outRow : 1;
            const int maxDown = (outRow + 1) > inputMinOne ? (inputMinOne - outRow) : 1;
            const int totalUpDown = maxUp + maxDown + 1;

            for (int outCol = 0; outCol < dim.outputSize; outCol++)
            {
                const int maxLeft = (outCol - 1) < 0 ? outCol : 1;
                const int maxRight = (outCol + 1) > inputMinOne ? (inputMinOne - outCol) : 1;
                const int totalLeftRight = maxLeft + maxRight + 1;

                sum = startSum;
                __m128 vSum = _mm_setzero_ps();
                __m256 vSum256 = _mm256_setzero_ps();

                if (totalLeftRight == 3 && totalUpDown == 3)
                {
                    const int inRow = outRow - 1;
                    const int inCol = outCol - 1;

                    const int startWeightIndex = weightOffset * 3;
                    const int startLocation = ((inputOffset + inRow) * dim.inputSize + inCol);

                    float* startWeight = &weights[startWeightIndex];
                    float* startInput = &inputData[startLocation];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startWeight += weightStepSize, startInput += inputStepSize)
                    {
                        float* row2 = &startInput[dim.inputSize];
                        float* row3 = &startInput[dim.inputSize + dim.inputSize];

                        __m256 weightsData = _mm256_load_ps(startWeight);
                        __m256 input = _mm256_setr_ps(startInput[0], startInput[1], startInput[2], row2[0], row2[1], row2[2], row3[0], row3[1]);

                        __m256 mulResult = _mm256_mul_ps(weightsData, input);
                        vSum256 = _mm256_add_ps(mulResult, vSum256);

                        sum += startWeight[8] * row3[2];
                    }
                }
                else if (totalLeftRight == 2 && totalUpDown == 2)
                {
                    const int inRow = outRow - maxUp;
                    const int filterRow = -maxUp + 1;

                    const int inCol = outCol - maxLeft;
                    const int filterCol = -maxLeft + 1;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat2x2(startInput[0], startInput[dim.inputSize]), SIMD_128::LoadFloat2x2(startWeight[0], startWeight[3])), vSum);
                    }
                }
                else if (totalLeftRight == 3 && totalUpDown == 2)
                {
                    const int inRow = outRow - maxUp;
                    const int filterRow = -maxUp + 1;

                    const int inCol = outCol - 1;
                    const int filterCol = 0;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        __m128 w1 = SIMD_128::LoadFloat3(startWeight[0]);
                        __m128 w2 = SIMD_128::LoadFloat3(startWeight[3]);

                        __m128 i1 = SIMD_128::LoadFloat3(startInput[0]);
                        __m128 i2 = SIMD_128::LoadFloat3(startInput[dim.inputSize]);

                        vSum = SIMD_128::AddFloats(SIMD_128::AddFloats(SIMD_128::MultiplyFloats(i1, w1), SIMD_128::MultiplyFloats(i2, w2)), vSum);
                    }
                }
                else if (totalLeftRight == 2 && totalUpDown == 3)
                {
                    const int inRow = outRow - 1;
                    const int filterRow = 0;

                    const int inCol = outCol - maxLeft;
                    const int filterCol = -maxLeft + 1;

                    const int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                    const int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                    float* startInput = &inputData[startInputIndex];
                    float* startWeight = &weights[startWeightIndex];

                    for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                    {
                        __m128 i1 = SIMD_128::LoadFloat2x2(startInput[0], startInput[dim.inputSize]);
                        __m128 i2 = SIMD_128::LoadFloat2(startInput[dim.inputSize + dim.inputSize]);

                        __m128 w1 = SIMD_128::LoadFloat2x2(startWeight[0], startWeight[3]);
                        __m128 w2 = SIMD_128::LoadFloat2(startWeight[6]);

                        vSum = SIMD_128::AddFloats(SIMD_128::AddFloats(SIMD_128::MultiplyFloats(i1, w1), SIMD_128::MultiplyFloats(i2, w2)), vSum);
                    }
                }
                else
                {
                    for (int u = -maxUp; u <= maxDown; u++)
                    {
                        const int inRow = outRow + u;
                        const int filterRow = u + 1;

                        const int inCol = outCol - maxLeft;
                        const int filterCol = -maxLeft + 1;

                        int startInputIndex = ((inputOffset + inRow) * dim.inputSize + inCol);
                        int startWeightIndex = ((weightOffset + filterRow) * 3 + filterCol);

                        float* startWeight = &weights[startWeightIndex];
                        float* startInput = &inputData[startInputIndex];

                        if (totalLeftRight == 3)
                        {
                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat3(startInput[0]), SIMD_128::LoadFloat3(startWeight[0])), vSum);
                            }
                        }
                        else
                        {
                            for (int inPlane = 0; inPlane < dim.inputPlanes; inPlane++, startInput += inputStepSize, startWeight += weightStepSize)
                            {
                                vSum = SIMD_128::AddFloats(SIMD_128::MultiplyFloats(SIMD_128::LoadFloat2(startInput[0]), SIMD_128::LoadFloat2(startWeight[0])), vSum);
                            }
                        }
                    }
                }

                vSum = SIMD_128::AddFloats(_mm256_extractf128_ps(vSum256, 0), vSum);
                vSum = SIMD_128::AddFloats(_mm256_extractf128_ps(vSum256, 1), vSum);
                sum += SIMD_128::ToFloat(vSum);

                int outputIndex = (outputTemp
                    + outRow)
                    * dim.outputSize + outCol;

                output[outputIndex] = sum;
            }
        }
    });
}
