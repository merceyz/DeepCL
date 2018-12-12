// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include "conv/Forward1.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"
#include <string>
#include <fstream>
#include <streambuf>
#include "VS2017\Helper.h"

#include <Windows.h>
#include <locale>
#include <codecvt>

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Forward1::~Forward1() {
    delete kernel;
}
VIRTUAL void Forward1::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("Forward1::forward START");

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    if (dim.biased)
        kernel->input(biasWrapper);
    kernel->output(outputWrapper);

    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = min(globalSize, cl->getMaxWorkgroupSize());
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

    StatefulTimer::timeCheck("Forward1::forward END");
}
Forward1::Forward1(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
    std::string options = "";
    options += dim.buildOptionsString();
    
    LoadKernel("forward1.cpp", "convolve_imagecubes_float2");
}
