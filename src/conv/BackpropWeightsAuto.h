// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "EasyCL.h"
#include "activate/ActivationFunction.h"
#include "conv/BackpropWeights.h"
#include "conv/LayerDimensions.h"
#include "DeepCLDllExport.h"

using namespace std;

//inline float square(float value) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT BackpropWeightsAuto : public BackpropWeights
{
public:
    //    EasyCL *cl;
    //    LayerDimensions dim;
    //    ActivationFunction const*fn;

    int num;
    int *milliseconds;
    bool *valid;
    int *numTries; // how many times did we try this kernel? ignore the first time, since opencl compilation time would be included
    int chosenIndex;
    BackpropWeights **instances;
    int currentIndex;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackpropWeightsAuto(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackpropWeightsAuto();
    VIRTUAL void calcGradWeights(
        int batchSize, CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput);

    // [[[end]]]

};


