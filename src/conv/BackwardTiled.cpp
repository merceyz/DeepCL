#include "BackwardTiled.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "VS2017\Helper.h"
using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackwardTiled::~BackwardTiled() {
    delete kernel;
//    delete broadcastMultiply;
//    delete applyActivationDeriv;
}
VIRTUAL void BackwardTiled::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("BackwardTiled start");

    kernel
       ->in(batchSize)
        ->in(gradOutputWrapper)
       ->in(weightsWrapper)
        ->out(gradInputWrapper);

	//ToDo
	int workgroupsize = 64;
	const int WAVE_SIZE = 64; 	//max of AMD: 64 and NV:  32 
	int workgroupPerImage =1;
	if (dim.inputSize > 19)
	{
		workgroupsize = 256;
		if (dim.filterSize == 3 && dim.inputSize > 32)
		{
			//16x64 tile
			const int HSIZE = 16;
			const int VSIZE = 64;
			workgroupPerImage = ((dim.inputSize + HSIZE - 1) / HSIZE) * ((dim.inputSize + VSIZE - 1) / VSIZE);
		}
		else
		{
			//32x32 tile
			const int HSIZE = 32;
			const int VSIZE = 32;
			workgroupPerImage = ((dim.inputSize + HSIZE - 1) / HSIZE) * ((dim.inputSize + VSIZE - 1) / VSIZE);
		}
	}
	else
	{
		int workgroupsize = (dim.inputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (workgroupsize < WAVE_SIZE)
			workgroupsize = WAVE_SIZE;
		workgroupPerImage = 1;
	}


	int globalSize = batchSize * dim.numInputPlanes * workgroupPerImage * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();
    StatefulTimer::instance()->timeCheck("BackwardTiled after first kernel");

//    applyActivationDeriv->in(batchSize * dim.inputCubeSize)->in(gradInputWrapper)->in(inputDataWrapper);
//    applyActivationDeriv->run_1d(globalSize, workgroupsize);
//    cl->finish();
//    StatefulTimer::instance()->timeCheck("BackwardTiled after applyActivationDeriv");
    
    StatefulTimer::instance()->timeCheck("BackwardTiled end");
}
BackwardTiled::BackwardTiled(EasyCL *cl, LayerDimensions dim) :
        Backward(cl, dim)
            {
    std::string options = dim.buildOptionsString();
    options += ""; // " -D " + upstreamFn->getDefineName();

	//ToDo
	if (dim.inputSize > 19)
	{
		if (dim.filterSize == 3 && dim.inputSize > 32)
		{   //16x64 tile
			options += " -D TILE_WIDTH=16";
			options += " -D TILE_HEIGHT=16";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}
		else
		{   //32x32 tile: LDS bank = 32
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
		int FIXED_WORKGROUP_SIZE = (dim.inputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if(FIXED_WORKGROUP_SIZE < 64)
			FIXED_WORKGROUP_SIZE = 64;
		int VTILE_HEGIHT = FIXED_WORKGROUP_SIZE / dim.inputSize;
		int VTILE_REPEAT = (dim.inputSizeSquared + FIXED_WORKGROUP_SIZE - 1) / FIXED_WORKGROUP_SIZE;

		options += " -D TILE_WIDTH=" + toString(dim.inputSize);
		options += " -D TILE_HEIGHT=" + toString(VTILE_HEGIHT);
		options += " -D VTILE_REPEAT=" + toString(VTILE_REPEAT);
		options += " -D FIXED_WORKGROUP_SIZE=" + toString(FIXED_WORKGROUP_SIZE);
	}

    LoadKernel("backwardTiled.cl", "calcGradInput_TileMode");
}
