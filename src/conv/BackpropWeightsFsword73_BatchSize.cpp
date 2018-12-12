// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsFsword73_BatchSize.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"
#include "VS2017\Helper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsFsword73_BatchSize::~BackpropWeightsFsword73_BatchSize() {
//    cout << "~backpropgradWeights2naive: deleting kernel" << endl;
    delete kernel;
}
VIRTUAL void BackpropWeightsFsword73_BatchSize::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsFsword73_BatchSize  start");
//     cout << "BackpropWeightsFsword73_BatchSize::calcGradWeights start ...." << endl;
    const float learningMultiplier = learningRateToMultiplier(batchSize);

    kernel
       ->in(batchSize)
       ->in(learningMultiplier)
        ->in(imagesWrapper)
       ->in(gradOutputWrapper)
       ->inout(gradWeightsWrapper);
    if(dim.biased) {
        kernel->inout(gradBiasWrapper);
    }

    const size_t workgroupSize = 64;
    const size_t numWorkgroups = (size_t)dim.numFilters * (size_t)dim.inputPlanes * (size_t)square(dim.filterSize);
    const size_t globalSize = workgroupSize * numWorkgroups;

    kernel->run(1, &globalSize, &workgroupSize);

    cl->finish();
//     cout << "... BackpropWeightsFsword73_BatchSize ::calcGradWeights done" << endl;	

    StatefulTimer::instance()->timeCheck("BackpropWeightsFsword73_BatchSize  end");

	//cout <<" dim.numFilters " << dim.numFilters << " dim.inputPlanes " << dim.inputPlanes  <<  " dim.filterSize " << dim.filterSize << " dim.outputSize "<< dim.outputSize << endl;
}
BackpropWeightsFsword73_BatchSize::BackpropWeightsFsword73_BatchSize(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    std::string options = dim.buildOptionsString();

    LoadKernel("fsword73_backpropweights_fast_batchSize.cl", "test_kernel");
}
