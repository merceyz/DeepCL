// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ForwardTiled.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"
#include "VS2017\Helper.h"


using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardTiled::~ForwardTiled() {
    delete kernel;
    delete addBias;
}
VIRTUAL void ForwardTiled::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardTiled::forward START");

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);

	//ToDo
	int workgroupsize =64;
	const int WAVE_SIZE = 64; 	//max of AMD: 64 and NV:  32 
	int workgroupPerImage =1;
	if (dim.outputSize > 19)
	{		
		workgroupsize = 256;
		if (dim.filterSize == 3 && dim.outputSize > 32)
		{
			//16x64 tile
			const int HSIZE = 16;
			const int VSIZE = 64;
			workgroupPerImage = ((dim.outputSize + HSIZE - 1) / HSIZE) * ((dim.outputSize + VSIZE - 1) / VSIZE);
		}
		else
		{ 
			//32x32 tile
			const int HSIZE = 32;
			const int VSIZE = 32;
			workgroupPerImage = ((dim.outputSize + HSIZE - 1) / HSIZE) * ((dim.outputSize + VSIZE - 1) / VSIZE);
		}
	}
	else
	{
		int workgroupsize = (dim.outputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (workgroupsize < 64)
			workgroupsize = 64;
		workgroupPerImage = 1;
	}


    int globalSize = batchSize * dim.numFilters * workgroupPerImage * workgroupsize;    
 	//    cout << "ForwardTiled globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardTiled::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("ForwardTiled::forward END");
}
ForwardTiled::ForwardTiled(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
    addBias = new AddBias(cl);

    std::string options = "";
    options += dim.buildOptionsString();

	//ToDo
	if (dim.outputSize > 19)
	{
		if(dim.filterSize == 3 && dim.outputSize > 32)
		{   //16x64 tile
			options += " -D TILE_WIDTH=16";
			options += " -D TILE_HEIGHT=16";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}
		else
		{   //32x32 tile
			options += " -D TILE_WIDTH=32";
			options += " -D TILE_HEIGHT=8";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}

	}
	else
	{
		const int WAVE_SIZE = 64; 
		//max of AMD: 64 and NV:  32 
		int FIXED_WORKGROUP_SIZE = (dim.outputSizeSquared /2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (FIXED_WORKGROUP_SIZE < 64)
			FIXED_WORKGROUP_SIZE = 64;
		int VTILE_HEGIHT = FIXED_WORKGROUP_SIZE / dim.outputSize;
		int VTILE_REPEAT = (dim.outputSizeSquared + FIXED_WORKGROUP_SIZE - 1) / FIXED_WORKGROUP_SIZE;

		options += " -D TILE_WIDTH=" + toString(dim.outputSize);
		options += " -D TILE_HEIGHT=" + toString(VTILE_HEGIHT);
		options += " -D VTILE_REPEAT=" + toString(VTILE_REPEAT);
		options += " -D FIXED_WORKGROUP_SIZE=" + toString(FIXED_WORKGROUP_SIZE);
	}

    LoadKernel("forwardTiled.cl", "convolve_tilemode_float");
}
