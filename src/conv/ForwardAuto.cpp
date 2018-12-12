// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "conv/ForwardAuto.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "util/Timer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

#define DeleteArray(a) if (a != 0) { delete[] a; a = 0; }

ForwardAuto::ForwardAuto(EasyCL *cl, LayerDimensions dim) :
	Forward(cl, dim),
	milliseconds(0),
	valid(0),
	chosenIndex(-1),
	instances(0)
{
	num = Forward::getNumImplementations();
	milliseconds = new int[num];
	valid = new bool[num];
	instances = new Forward *[num];
	for (int i = 0; i < num; i++)
	{
		instances[i] = 0;
		valid[i] = false;
		milliseconds[i] = -1;
	}
	nextIndex = 0;
}

VIRTUAL ForwardAuto::~ForwardAuto()
{
    if (ChosenInstance != 0)
    {
        delete ChosenInstance;
        ChosenInstance = 0;
    }

    if (instances != 0)
    {
        for (int i = 0; i < num; i++)
        {
            if (instances[i] != 0)
            {
                delete instances[i];
            }
        }

        delete[] instances;
    }

    DeleteArray(milliseconds);
    DeleteArray(valid);
}
VIRTUAL void ForwardAuto::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper,
	CLWrapper *biasWrapper, CLWrapper *outputWrapper)
{
	//    Forward *instance = 0;
	//    cout << "ForwardAuto::forward" << endl;

    if (num == 1)
    {
        const int kernelTesting = 0;
        instances[0] = Forward::instanceSpecific(batchSize, kernelTesting, cl, dim);
        Timer timer;
        instances[0]->forward(batchSize, dataWrapper, weightsWrapper, biasWrapper, outputWrapper);
        const int timeTaken = (int)timer.elapsedMicroseconds();
        if (milliseconds[0] > timeTaken || milliseconds[0] == -1)
            milliseconds[0] = timeTaken;

        cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << kernelTesting << " " << milliseconds[0] << " microseconds" << endl;
        chosenIndex = kernelTesting;
        return;
    }

	while (chosenIndex == -1 && nextIndex < num)
	{
		int thisIndex = nextIndex;
		nextIndex++;
		cout << "forward try kernel " << thisIndex << endl;
		if (Forward::plausiblyOptimal(thisIndex, batchSize, dim))
		{
			Forward *candidate = 0;
			try
			{
				candidate = Forward::instanceSpecific(batchSize, thisIndex, cl, dim);
				instances[thisIndex] = candidate;
				valid[thisIndex] = true;
				cout << "   ... seems valid" << endl;
			}
			catch (runtime_error &e)
			{
				cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << ": this instance cant be used: " << e.what() << endl;
				valid[thisIndex] = false;
                delete instances[thisIndex];
                instances[thisIndex] = 0;
			}
			if (valid[thisIndex])
			{
				Timer timer;
				try
				{
					candidate->forward(batchSize, dataWrapper, weightsWrapper, biasWrapper, outputWrapper);
					milliseconds[thisIndex] = (int)timer.elapsedMicroseconds();
					cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << " " << milliseconds[thisIndex] << " microseconds" << endl;
					return;
				}
				catch (runtime_error &e)
				{
					cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << " this instance cant be used: " << e.what() << endl;
					valid[thisIndex] = false;
					delete instances[thisIndex];
					instances[thisIndex] = 0;
				}
			}
			else
			{
				cout << "   ... not valid" << endl;
			}
		}
		else
		{
			cout << "  ... not plausibly optimal, skipping" << endl;
		}
	}
	if (chosenIndex == -1)
	{
		//        cout << StatefulTimer::instance()->prefix + "ForwardAuto::forward choosing best instance:" << endl;
		int bestIndex = -1;
		int bestTime = 0;
		for (int i = 0; i < num; i++)
		{
			if (!valid[i])
			{
				cout << "   forward kernel " << i << ": cannot be used" << endl;
				continue;
			}
			cout << "   forward kernel " << i << " time: " << milliseconds[i] << " microseconds" << endl;
			if (bestIndex == -1)
			{
				bestIndex = i;
				bestTime = milliseconds[i];
				continue;
			}
			if (milliseconds[i] < bestTime)
			{
				bestTime = milliseconds[i];
				bestIndex = i;
			}
		}
		if (bestIndex != -1)
		{
			cout << "   forward layer selected kernel " << bestIndex << endl;
			this->chosenIndex = bestIndex;
		}
		else
		{
			throw runtime_error(StatefulTimer::instance()->prefix + "No valid forward implementations found");
		}
	}

	if (chosenIndex != -1 && ChosenInstance == 0)
	{
        if (instances[chosenIndex] == 0)
        {
            ChosenInstance = Forward::instanceSpecific(batchSize, chosenIndex, cl, dim);
        }
        else
        {
            ChosenInstance = instances[chosenIndex];
            instances[chosenIndex] = 0;
        }

        for (int i = 0; i < num; i++)
        {
            if (instances[i] != 0)
                delete instances[i];
        }

        DeleteArray(instances);
        DeleteArray(milliseconds);
        DeleteArray(valid);
	}

	//    cout << "ForwardAuto::forward using instance index: " << chosenIndex << endl;
    ChosenInstance->forward(batchSize, dataWrapper, weightsWrapper, biasWrapper, outputWrapper);
}
