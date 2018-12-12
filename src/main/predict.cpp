// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once
#include "DeepCL.h"
#include "loss/SoftMaxLayer.h"
#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
#include "clblas/ClBlasInstance.h"
#include "conv\ForwardAuto.h"
#include "fc\FullyConnectedLayer.h"
#include <Windows.h>
#include <random>
#include "VS2017\HelperClass.h"


#define KillProcess                                                                         \
{                                                                                           \
    HANDLE hnd;                                                                             \
    hnd = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, TRUE, GetCurrentProcessId());        \
    TerminateProcess(hnd, 0);                                                               \
}                                                                                           \

using namespace std;

/* [[[cog
	# These are used in the later cog sections in this file:
	options = [
		{'name': 'gpuIndex', 'type': 'int', 'description': 'gpu device index; default value is gpu if present, cpu otw.', 'default': -1, 'ispublicapi': True},

		{'name': 'weightsFile', 'type': 'string', 'description': 'file to read weights from', 'default': 'weights.dat', 'ispublicapi': True},
		# removing loadondemand for now, let's always load exactly one batch at a time for now
		# ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, [0,1], True},
		{'name': 'batchSize', 'type': 'int', 'description': 'batch size', 'default': 128, 'ispublicapi': True},

		# lets go with pipe for now, and then somehow shoehorn files in later?
		{'name': 'inputFile',  'type': 'string', 'description': 'file to read inputs from, if empty, read stdin (default)', 'default': ''},
		{'name': 'outputFile', 'type': 'string', 'description': 'file to write outputs to, if empty, write to stdout', 'default': ''},
		{'name': 'outputLayer', 'type': 'int', 'description': 'layer to write output from, default -1 means: last layer', 'default': -1},
		{'name': 'writeLabels', 'type': 'int', 'description': 'write integer labels, instead of probabilities etc (default 0)', 'default': 0},
		{'name': 'outputFormat', 'type': 'string', 'description': 'output format [binary|text]', 'default': 'text'}
	]
*///]]]
// [[[end]]]

class Config
{
public:
	/* [[[cog
		cog.outl('// generated using cog:')
		for option in options:
			cog.outl(option['type'] + ' ' + option['name'] + ';')
	*/// ]]]
	// generated using cog:
	int gpuIndex;
	string weightsFile;
	int batchSize;
	string inputFile;
	string outputFile;
	int outputLayer;
	int writeLabels;
	string outputFormat;
	string kernelCacheFile;
    bool profileRun;
    bool exitInstantly;
	// [[[end]]]

	Config()
	{
		/* [[[cog
			cog.outl('// generated using cog:')
			for option in options:
				defaultString = ''
				default = option['default']
				type = option['type']
				if type == 'string':
					defaultString = '"' + default + '"'
				elif type == 'int':
					defaultString = str(default)
				elif type == 'float':
					defaultString = str(default)
					if '.' not in defaultString:
						defaultString += '.0'
					defaultString += 'f'
				cog.outl(option['name'] + ' = ' + defaultString + ';')
		*/// ]]]
		// generated using cog:
		gpuIndex = -1;
		weightsFile = "weights.dat";
		batchSize = 128;
		inputFile = "";
		outputFile = "";
		kernelCacheFile = "";
		outputLayer = -1;
		writeLabels = 0;
		outputFormat = "text";
        profileRun = false;
        exitInstantly = true;
		// [[[end]]]
	}
};

void go(Config config)
{
	int N = -1;
	int numPlanes;
	int imageSize;
	int imageSizeCheck;

	int dims[3];
    if (config.profileRun == true)
    {
        dims[0] = 3;
        dims[1] = 96;
        dims[2] = 96;
    }
    else
	    cin.read(reinterpret_cast<char *>(dims), 3 * 4l);

	numPlanes = dims[0];
	imageSize = dims[1];
	imageSizeCheck = dims[2];
	if (imageSize != imageSizeCheck)
	{
		throw std::runtime_error("imageSize doesnt match imageSizeCheck, image not square");
	}

	const int64 inputCubeSize = numPlanes * imageSize * imageSize;

	//
	// ## Set up the Network
	//

	EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu(false);

	NeuralNet *net = new NeuralNet(cl);

	// just use the default for net creation, weights are overriden from the weightsFile
	WeightsInitializer *weightsInitializer = new OriginalInitializer();

	string netDef;
	if (!WeightsPersister::loadConfigString(config.weightsFile, netDef))
	{
		cout << "Cannot load network definition from weightsFile." << endl;
		return;
	}

	net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
	net->addLayer(NormalizationLayerMaker::instance()->translate(0.0f)->scale(1.0f)); // This will be read from weights file

	Timer time;
	if (!NetdefToNet::createNetFromNetdef(net, netDef, weightsInitializer))
	{
		return;
	}

	// ignored int and float, s.t. we can use loadWeights
	int ignI;
	float ignF;

	// weights file contains normalization layer parameters as 'weights' now.  We should probably rename weights to parameters
	// sooner or later ,but anyway, tehcnically, works for onw
	if (!WeightsPersister::loadWeights(config.weightsFile, string("netDef=") + netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF))
	{
		cout << "Cannot load network weights from weightsFile." << endl;
		return;
	}

	net->setBatchSize(config.batchSize);

#pragma region Load kernel cache
	bool hasLoadedCache = false;
	try
	{
		if (config.kernelCacheFile.length() > 0)
		{
			ifstream loadConf;
			loadConf.open(config.kernelCacheFile);

			if (loadConf.good())
			{
				string line = "";
				while (std::getline(loadConf, line))
				{
					if (line.empty() == false && line.length() > 0)
					{
						vector<string> tempData = split(line, "|");

						Layer *layer = net->getLayer(stoi(tempData[0]));
						string name = layer->getClassName();

						if (name == "ConvolutionalLayer")
						{
							ConvolutionalLayer *conv = dynamic_cast<ConvolutionalLayer *>(layer);
							ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(conv->forwardImpl);
							forwardAuto->chosenIndex = stoi(tempData[1]);
						}
						else if (name == "FullyConnectedLayer")
						{
							FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer *>(layer);
							ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(fc->convolutionalLayer->forwardImpl);
							forwardAuto->chosenIndex = stoi(tempData[1]);
						}

						hasLoadedCache = true;
						cout << "Loaded cached kernel values for layer " << tempData[0] << endl;
					}
				}
				loadConf.close();
			}
		}
	}
	catch (const std::exception&)
	{

	}
#pragma endregion

#pragma region Print out layer dim
	for (int i = 0; i < net->getNumLayers(); i++)
	{
		Layer *layer = net->getLayer(i);
		string name = layer->getClassName();

		if (name == "ConvolutionalLayer")
		{
			ConvolutionalLayer *conv = dynamic_cast<ConvolutionalLayer *>(layer);
			ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(conv->forwardImpl);
			cout << forwardAuto->dim << endl;
		}
		else if (name == "FullyConnectedLayer")
		{
			FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer *>(layer);
			ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(fc->convolutionalLayer->forwardImpl);
			cout << forwardAuto->dim << endl;
		}
	}
#pragma endregion

	//
	// ## All is set up now
	//

	float *inputData = new float[inputCubeSize * config.batchSize];
	bool *resultData = new bool[config.batchSize];
	float *floatData = new float[config.batchSize];

	int *batchData = new int[2]
	{
		config.batchSize, // Current BatchSize
		config.batchSize  // Previous batch size
	};

	char *readBuffer = new char[4];

	// Send the pointer of the input data to the stream
	cout << "LOC: " << std::addressof(inputData[0]) << endl;
	cout << "LOC: " << std::addressof(resultData[0]) << endl;
	cout << "LOC: " << std::addressof(floatData[0]) << endl;
	cout << "LOC: " << std::addressof(batchData[0]) << endl;

	bool more = true;

	if (config.outputLayer == -1)
	{
		config.outputLayer = net->getNumLayers() - 1;
	}

    InputLayer *directInput = dynamic_cast<InputLayer *>(net->getLayer(0));
    const int numFields = net->getLayer(config.outputLayer)->getOutputCubeSize();

    if (config.profileRun)
    {
        for (int rounds = 0; rounds < 100; rounds++)
        {
            float counter = 0;
            for (int i = 0; i < inputCubeSize * config.batchSize; i++)
            {
                inputData[i] = counter += .1f;
            }

            net->setBatchSize(batchData[0]);

            directInput->in(inputData);

            for (int layerId = 0; layerId <= config.outputLayer; layerId++)
            {
                net->getLayer(layerId)->forward();
            }

            cout << endl;

            float const *output = net->getLayer(config.outputLayer)->getOutput();

            for (int i = 0; i < batchData[0]; i++)
            {
                float res = output[(i * numFields) + 1];
                floatData[i] = res;
            }
        }

        ofstream write;
        write.open("E:\\Programming\\Testing stuff\\ConsoleApp1\\ConsoleApp1\\bin\\Debug\\Output.txt", std::ios_base::app);
        for (int layerId = 0; layerId < net->getNumLayers(); layerId++)
        {
            Layer *layer = net->getLayer(layerId);
            string name = layer->getClassName();

            if (name == "ConvolutionalLayer" || name == "FullyConnectedLayer")
            {
                ConvolutionalLayer *conv = 0;
                if (name == "ConvolutionalLayer")
                    conv = dynamic_cast<ConvolutionalLayer *>(layer);
                else
                {
                    FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer *>(layer);
                    conv = fc->convolutionalLayer;
                }

                ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(conv->forwardImpl);

                if (forwardAuto->milliseconds != 0)
                    write << forwardAuto->milliseconds[0] << endl;
            }
        }
        
        cout << endl << "Result" << endl;

        ifstream in(Helper::GetStartupPath() + "\\Results.bin", ios::binary, SH_DENYNO);
        ofstream out;
        
        byte* buffer = new byte[sizeof(float)];

        int missmatch = 0;
        bool firstTime = true;
        for (int i = 0; i < batchData[0]; i++)
        {
            if (in.good())
                in.read((char*)buffer, sizeof(float));

            if (in.good() && in.gcount() == sizeof(float))
            {
                if (fabs(floatData[i] - ((float*)buffer)[0]) > 1e-8f)
                {
                    missmatch++;

                    cout << floatData[i] << ", should be " << ((float*)buffer)[0] << ", difference: " << fabs(floatData[i] - ((float*)buffer)[0]) << endl;
                }
                else cout << floatData[i] << endl;
            }
            else
            {
                if (firstTime)
                {
                    in.close();
                    firstTime = false;
                    out.open(Helper::GetStartupPath() + "\\Results.bin", ios::binary, SH_DENYNO);
                    i = -1;
                    continue;
                }

                auto x = reinterpret_cast<const char*>(&floatData[i]);
                out.write(x, sizeof(float));
                cout << endl << floatData[i] << " dumped to file";
            }
        }

        cout << endl << endl << "Test finished, missmatch count: " << missmatch << endl;

        delete[] buffer;
    }
    else
    {
        cin.read(readBuffer, 4);
        more = !cin.eof();

        while (more)
        {
            if (batchData[1] != batchData[0])
            {
                net->setBatchSize(batchData[0]);
                batchData[1] = batchData[0];
            }

            directInput->in(inputData);
            for (int layerId = 0; layerId <= config.outputLayer; layerId++)
            {
                net->getLayer(layerId)->forward();
            }

            float const *output = net->getLayer(config.outputLayer)->getOutput();

            for (int i = 0; i < batchData[0]; i++)
            {
                float res = output[(i * numFields) + 1];
                floatData[i] = res;
                if (res >= .5)
                    resultData[i] = true;
                else resultData[i] = false;
            }

            cout << "END" << endl;

            cin.read(readBuffer, 4);
            more = !cin.eof();
        }

        #pragma region Save kernel cache if it doesnt already exist
        try
        {
            if (config.kernelCacheFile.length() > 0 && hasLoadedCache == false)
            {
                ofstream cacheFile;
                cacheFile.open(config.kernelCacheFile);

                for (int layerId = 0; layerId < net->getNumLayers(); layerId++)
                {
                    Layer *layer = net->getLayer(layerId);
                    string name = layer->getClassName();

                    if (name == "ConvolutionalLayer" || name == "FullyConnectedLayer")
                    {
                        cacheFile << toString(layerId);

                        ConvolutionalLayer *conv = 0;
                        if (name == "ConvolutionalLayer")
                            conv = dynamic_cast<ConvolutionalLayer *>(layer);
                        else
                        {
                            FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer *>(layer);
                            conv = fc->convolutionalLayer;
                        }

                        ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(conv->forwardImpl);

                        if (forwardAuto == 0)
                            cacheFile << "|-1" << endl;
                        else
                            cacheFile << "|" << toString(forwardAuto->chosenIndex) << endl;
                    }
                }

                cacheFile.close();
            }
        }
        catch (const std::exception&)
        {

        }
        #pragma endregion
    }

	delete[] inputData;
	delete[] resultData;
	delete[] floatData;
	delete[] readBuffer;
	delete[] batchData;
	delete weightsInitializer;
	delete net;
	delete cl;
}

void printUsage(char *argv[], Config config)
{
	cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
	cout << endl;
	cout << "Possible key=value pairs:" << endl;
	/* [[[cog
		cog.outl('// generated using cog:')
		cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
		for option in options:
			name = option['name']
			description = option['description']
			if 'ispublicapi' in option and option['ispublicapi']:
				cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
		cog.outl('cout << "" << endl; ')
		cog.outl('cout << "unstable, might change within major version:" << endl; ')
		for option in options:
			if 'ispublicapi' not in option or not option['ispublicapi']:
				name = option['name']
				description = option['description']
				cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
	*///]]]
	// generated using cog:
	cout << "public api, shouldnt change within major version:" << endl;
	cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
	cout << "    weightsfile=[file to read weights from] (" << config.weightsFile << ")" << endl;
	cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
	cout << "" << endl;
	cout << "unstable, might change within major version:" << endl;
	cout << "    inputfile=[file to read inputs from, if empty, read stdin (default)] (" << config.inputFile << ")" << endl;
	cout << "    outputfile=[file to write outputs to, if empty, write to stdout] (" << config.outputFile << ")" << endl;
	cout << "    outputlayer=[layer to write output from, default -1 means: last layer] (" << config.outputLayer << ")" << endl;
	cout << "    writelabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeLabels << ")" << endl;
	cout << "    outputformat=[output format [binary|text]] (" << config.outputFormat << ")" << endl;
	// [[[end]]]
}


LONG WINAPI OurCrashHandler(EXCEPTION_POINTERS * ExceptionInfo)
{
    /*if (dontCreateDump == false && ExceptionInfo != nullptr)
    {
    CreateDirectory(L"\\Crash dumps", NULL);
    make_minidump(ExceptionInfo);
    }*/

    KillProcess;
    return false ? EXCEPTION_CONTINUE_SEARCH : EXCEPTION_EXECUTE_HANDLER;
}

void terimateHandler()
{
    KillProcess;
}

int main(int argc, char *argv[])
{
    ::SetUnhandledExceptionFilter(OurCrashHandler);

    set_unexpected(terimateHandler);
    set_terminate(terimateHandler);

    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);

	Config config;
	if (argc == 2 && (string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h"))
	{
		printUsage(argv, config);
	}
	for (int i = 1; i < argc; i++)
	{
		vector<string> splitkeyval = split(argv[i], "=");
		if (splitkeyval.size() != 2)
		{
			cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
			exit(1);
		}
		else
		{
			string key = splitkeyval[0];
			string value = splitkeyval[1];
			//            cout << "key [" << key << "]" << endl;
						/* [[[cog
							cog.outl('// generated using cog:')
							cog.outl('if(false) {')
							for option in options:
								name = option['name']
								type = option['type']
								cog.outl('} else if(key == "' + name.lower() + '") {')
								converter = '';
								if type == 'int':
									converter = 'atoi';
								elif type == 'float':
									converter = 'atof';
								cog.outl('    config.' + name + ' = ' + converter + '(value);')
						*/// ]]]
						// generated using cog:
			if (false)
			{
			}
			else if (key == "gpuindex")
			{
				config.gpuIndex = atoi(value);
			}
			else if (key == "weightsfile")
			{
				config.weightsFile = (value);
			}
			else if (key == "batchsize")
			{
				config.batchSize = atoi(value);
			}
			else if (key == "inputfile")
			{
				config.inputFile = (value);
			}
			else if (key == "outputfile")
			{
				config.outputFile = (value);
			}
			else if (key == "outputlayer")
			{
				config.outputLayer = atoi(value);
			}
			else if (key == "writelabels")
			{
				config.writeLabels = atoi(value);
			}
			else if (key == "outputformat")
			{
				config.outputFormat = (value);
			}
			else if (key == "kernelCacheFile")
			{
				config.kernelCacheFile = (value);
			}
            else if (key == "PROFILE")
            {
                config.profileRun = value == "true" ? true : false;

                if (config.profileRun)
                    StatefulTimer::instance()->setEnabled(true);
            }
            else if (key == "HOLDEXIT")
            {
                config.exitInstantly = value == "true" ? false : true;
            }
			else
			{
				cout << endl;
				cout << "Error: key '" << key << "' not recognised" << endl;
				cout << endl;
				printUsage(argv, config);
				cout << endl;
				return -1;
			}
		}
	}
	if (config.outputFormat != "text" && config.outputFormat != "binary")
	{
		cout << endl;
		cout << "outputformat must be 'text' or 'binary'" << endl;
		cout << endl;
		return -1;
	}

    int returnCode = 0;
	try
	{
		go(config);
	}
	catch (runtime_error e)
	{
		cout << "Something went wrong: " << e.what() << endl;
        returnCode = -1;
	}

    if (config.profileRun && config.exitInstantly == false)
    {
        char* buffer = new char[1];
        cin.read(buffer, 1);
        delete[] buffer;
    }

    return returnCode;
}