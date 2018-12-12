// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsFsword73.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"
#include "VS2017\Helper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsFsword73::~BackpropWeightsFsword73() {
//    cout << "~backpropgradWeights2naive: deleting kernel" << endl;
    delete kernel;
}
VIRTUAL void BackpropWeightsFsword73::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsFsword73 start");
//     cout << "BackpropWeightsFsword73::calcGradWeights start ...." << endl;
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
    const size_t numWorkgroups = (size_t)batchSize * (size_t)dim.numFilters * (size_t)dim.inputPlanes * (size_t)square(dim.filterSize);
    const size_t globalSize = workgroupSize * numWorkgroups;

    kernel->run(1, &globalSize, &workgroupSize);

    cl->finish();
//     cout << "... BackpropWeightsFsword73::calcGradWeights done" << endl;

    StatefulTimer::instance()->timeCheck("BackpropWeightsFsword73 end");
}
BackpropWeightsFsword73::BackpropWeightsFsword73(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    std::string options = dim.buildOptionsString();

    LoadKernel("fsword73_backpropweights_fast.cl", "test_kernel");
}
