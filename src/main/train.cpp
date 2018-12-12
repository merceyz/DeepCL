// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


//#include <iostream>
//#include <algorithm>

#include "DeepCL.h"
//#include "test/Sampler.h"  // TODO: REMOVE THIS
#include "clblas/ClBlasInstance.h"
#include "conv\ForwardAuto.h"
#include "conv/BackwardAuto.h"
#include "conv/BackpropWeightsAuto.h"
#include "fc\FullyConnectedLayer.h"

using namespace std;

/* [[[cog
	# These are used in the later cog sections in this file:
	# format:
	# (name, type, description, default, ispublicapi)
	options = [
		('gpuIndex', 'int', 'gpu device index; default value is gpu if present, cpu otw.', -1, True),
		('dataDir', 'string', 'directory to search for train and validate files', '../data/mnist', True),
		('trainFile', 'string', 'path to training data file',"train-images-idx3-ubyte", True),
		('dataset', 'string', 'choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]','', True),
		('validateFile', 'string', 'path to validation data file',"t10k-images-idx3-ubyte", True),
		('numTrain', 'int', 'num training examples',-1, True),
		('numTest', 'int', 'num test examples]',-1, True),
		('batchSize', 'int', 'batch size',128, True),
		('numEpochs', 'int', 'number epochs',12, True),
		('netDef', 'string', 'network definition',"rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n", True),
		('loadWeights', 'int', 'load weights from file at startup?', 0, True),
		('weightsFile', 'string', 'file to write weights to','weights.dat', True),
		('writeWeightsInterval', 'float', 'write weights every this many minutes', 0, True),
		('normalization', 'string', '[stddev|maxmin]', 'stddev', True),
		('normalizationNumStds', 'float', 'with stddev normalization, how many stddevs from mean is 1?', 2.0, True),
		('dumpTimings', 'int', 'dump detailed timings each epoch? [1|0]', 0, True),
		('multiNet', 'int', 'number of Mcdnn columns to train', 1, True),
		('loadOnDemand', 'int', 'load data on demand [1|0]', 0, True),
		('fileReadBatches', 'int', 'how many batches to read from file each time? (for loadondemand=1)', 50, True),
		('normalizationExamples', 'int', 'number of examples to read to determine normalization parameters', 10000, True),
		('weightsInitializer', 'string', 'initializer for weights, choices: original, uniform (default: original)', 'original', True),
		('initialWeights', 'float', 'for uniform initializer, weights will be initialized randomly within range -initialweights to +initialweights, divided by fanin, (default: 1.0f)', 1.0, False),
		('trainer', 'string', 'which trainer, sgd, anneal, nesterov, adagrad, rmsprop, or adadelta (default: sgd)', 'sgd', True),
		('learningRate', 'float', 'learning rate, a float value, used by all trainers', 0.002, True),
		('rho', 'float', 'rho decay, in adadelta trainer. 1 is no decay. 0 is full decay (default 0.9)', 0.9, False),
		('momentum', 'float', 'momentum, used by sgd and nesterov trainers', 0.0, True),
		('weightDecay', 'float', 'weight decay, 0 means no decay; 1 means full decay, used by sgd trainer', 0.0, True),
		('anneal', 'float', 'multiply learningrate by this amount each epoch, used by anneal trainer, default 1.0', 1.0, False)
	]
*///]]]
// [[[end]]]

class Config
{
public:
	/* [[[cog
		cog.outl('// generated using cog:')
		for (name,type,description,default,_) in options:
			cog.outl(type + ' ' + name + ';')
	*/// ]]]
	// generated using cog:
	int gpuIndex;
	string dataDir;
	string trainFile;
	string dataset;
	string validateFile;
	int numTrain;
	int numTest;
	int batchSize;
	int numEpochs;
	string netDef;
	int loadWeights;
	string weightsFile;
	float writeWeightsInterval;
	string normalization;
	float normalizationNumStds;
	int dumpTimings;
	int multiNet;
	int loadOnDemand;
	int fileReadBatches;
	int normalizationExamples;
	string weightsInitializer;
	float initialWeights;
	string trainer;
	float learningRate;
	float rho;
	float momentum;
	float weightDecay;
	float anneal;
	string kernelCacheFile;
	// [[[end]]]

	Config()
	{
		/* [[[cog
			cog.outl('// generated using cog:')
			for (name,type,description,default,_) in options:
				defaultString = ''
				if type == 'string':
					defaultString = '"' + default + '"'
				elif type == 'int':
					defaultString = str(default)
				elif type == 'float':
					defaultString = str(default)
					if '.' not in defaultString:
						defaultString += '.0'
					defaultString += 'f'
				cog.outl(name + ' = ' + defaultString + ';')
		*/// ]]]
		// generated using cog:
		gpuIndex = -1;
		dataDir = "";
		trainFile = "";
		dataset = "";
		validateFile = "";
		numTrain = -1;
		numTest = -1;
		batchSize = 128;
		numEpochs = 12;
		netDef = "rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n";
		loadWeights = 0;
		weightsFile = "weights.dat";
		kernelCacheFile = "";
		writeWeightsInterval = 0.0f;
		normalization = "stddev";
		normalizationNumStds = 2.0f;
		dumpTimings = 0;
		multiNet = 1;
		loadOnDemand = 0;
		fileReadBatches = 50;
		normalizationExamples = 10000;
		weightsInitializer = "original";
		initialWeights = 1.0f;
		trainer = "sgd";
		learningRate = 0.002f;
		rho = 0.9f;
		momentum = 0.0f;
		weightDecay = 0.0f;
		anneal = 1.0f;
		// [[[end]]]

	}
	string getTrainingString()
	{
		string configString = "";
		configString += "netDef=" + netDef; // lets just force that at least
				   // need same network structure, otherwise weights wont
				   // really make sense at all.  Evreything else is up to the
				   // end-user plausibly?
		return configString;
	}
	string getOldTrainingString()
	{
		string configString = "";
		configString += "netDef=" + netDef + " trainFile=" + trainFile;
		return configString;
	}
};

void go(Config config)
{
	Timer timer;

	int64 Ntrain = 0;
	int Ntest = 0;
	int numPlanes = 0;
	int imageSize = 0;

	float *trainData = 0;
	float *testData = 0;
	int *trainLabels = 0;
	int *testLabels = 0;

	if (config.dumpTimings)
	{
		StatefulTimer::setEnabled(true);
	}

	cout << std::addressof(Ntrain) << endl;
	cout << std::addressof(Ntest) << endl;
	cout << std::addressof(numPlanes) << endl;
	cout << std::addressof(imageSize) << endl;

	char *buffer = new char[4];
	cin.read(buffer, 4);
	std::string input = std::string(buffer);
	if (input != "STA")
	{
		cout << "Invalid input, \"" << input << "\"" << endl;
		return;
	}

	const int inputCubeSize = numPlanes * imageSize * imageSize;

	trainData = new float[Ntrain * inputCubeSize];
	trainLabels = new int[Ntrain];

	testData = new float[Ntest * inputCubeSize];
	testLabels = new int[Ntest];

	cout << std::addressof(trainData[0]) << endl;
	cout << std::addressof(trainLabels[0]) << endl;

	cout << std::addressof(testData[0]) << endl;
	cout << std::addressof(testLabels[0]) << endl;

	cin.read(buffer, 4);
	input = std::string(buffer);
	if (input != "STA")
	{
		cout << "Invalid input, \"" << input << "\"" << endl;
		return;
	}
	delete[] buffer;

	float translate;
	float scale;
	int normalizationExamples = config.normalizationExamples > Ntrain ? Ntrain : config.normalizationExamples;
	if (!config.loadOnDemand)
	{
		if (config.normalization == "stddev")
		{
			float mean, stdDev;
			NormalizationHelper::getMeanAndStdDev(trainData, normalizationExamples * inputCubeSize, &mean, &stdDev);
			cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
			translate = -mean;
			scale = 1.0f / stdDev / config.normalizationNumStds;
		}
		else if (config.normalization == "maxmin")
		{
			float mean, stdDev;
			NormalizationHelper::getMinMax(trainData, normalizationExamples * inputCubeSize, &mean, &stdDev);
			translate = -mean;
			scale = 1.0f / stdDev;
		}
		else
		{
			cout << "Error: Unknown normalization: " << config.normalization << endl;
			return;
		}
	}
	cout << " image norm translate " << translate << " scale " << scale << endl;
	timer.timeCheck("after getting stats");

	EasyCL *cl = 0;
	if (config.gpuIndex >= 0)
	{
		cl = EasyCL::createForIndexedGpu(config.gpuIndex);
	}
	else
	{
		cl = EasyCL::createForFirstGpuOtherwiseCpu();
	}
	ClBlasInstance blasInstance;

	NeuralNet *net;
	net = new NeuralNet(cl);

	WeightsInitializer *weightsInitializer = 0;
	if (toLower(config.weightsInitializer) == "original")
	{
		weightsInitializer = new OriginalInitializer();
	}
	else if (toLower(config.weightsInitializer) == "uniform")
	{
		weightsInitializer = new UniformInitializer(config.initialWeights);
	}
	else
	{
		cout << "Unknown weights initializer " << config.weightsInitializer << endl;
		return;
	}

	//    net->inputMaker<unsigned char>()->numPlanes(numPlanes)->imageSize(imageSize)->insert();
	net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
	net->addLayer(NormalizationLayerMaker::instance()->translate(translate)->scale(scale));
	if (!NetdefToNet::createNetFromNetdef(net, config.netDef, weightsInitializer))
	{
		return;
	}
	// apply the trainer
	Trainer *trainer = 0;
	if (toLower(config.trainer) == "sgd")
	{
		SGD *sgd = new SGD(cl);
		sgd->setLearningRate(config.learningRate);
		sgd->setMomentum(config.momentum);
		sgd->setWeightDecay(config.weightDecay);
		trainer = sgd;
	}
	else if (toLower(config.trainer) == "anneal")
	{
		Annealer *annealer = new Annealer(cl);
		annealer->setLearningRate(config.learningRate);
		annealer->setAnneal(config.anneal);
		trainer = annealer;
	}
	else if (toLower(config.trainer) == "nesterov")
	{
		Nesterov *nesterov = new Nesterov(cl);
		nesterov->setLearningRate(config.learningRate);
		nesterov->setMomentum(config.momentum);
		trainer = nesterov;
	}
	else if (toLower(config.trainer) == "adagrad")
	{
		Adagrad *adagrad = new Adagrad(cl);
		adagrad->setLearningRate(config.learningRate);
		trainer = adagrad;
	}
	else if (toLower(config.trainer) == "rmsprop")
	{
		Rmsprop *rmsprop = new Rmsprop(cl);
		rmsprop->setLearningRate(config.learningRate);
		trainer = rmsprop;
	}
	else if (toLower(config.trainer) == "adadelta")
	{
		Adadelta *adadelta = new Adadelta(cl, config.rho);
		trainer = adadelta;
	}
	else
	{
		cout << "trainer " << config.trainer << " unknown." << endl;
		return;
	}
	cout << "Using trainer " << trainer->asString() << endl;
	//    trainer->bindTo(net);
	//    net->setTrainer(trainer);
	net->setBatchSize(config.batchSize);
	net->print();

#pragma region Load kernel cache
	if (config.kernelCacheFile.length() > 0)
	{
		ifstream loadConf;
		loadConf.open(config.kernelCacheFile);

		if (loadConf.good())
		{
			string line = "";
			while (std::getline(loadConf, line))
			{
				if (line.length() != 0)
				{
					vector<string> tempData = split(line, "|");

					if (tempData.size() >= 2)
					{
						Layer *layer = net->getLayer(stoi(tempData[0]));
						string name = layer->getClassName();

						if (name == "ConvolutionalLayer" || name == "FullyConnectedLayer")
						{
							ConvolutionalLayer *conv;

							if (name == "ConvolutionalLayer")
								conv = dynamic_cast<ConvolutionalLayer *>(layer);
							else
							{
								FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer *>(layer);
								conv = fc->convolutionalLayer;
							}

							ForwardAuto *forwardAuto = dynamic_cast<ForwardAuto *>(conv->forwardImpl);
							BackwardAuto *backwardAuto = dynamic_cast<BackwardAuto *>(conv->backwardImpl);
							BackpropWeightsAuto *backpropWeightsAuto = dynamic_cast<BackpropWeightsAuto *>(conv->backpropWeightsImpl);

							if (forwardAuto != 0) forwardAuto->chosenIndex = stoi(tempData[1]);
							if (backwardAuto != 0 && tempData.size() >= 3) backwardAuto->chosenIndex = stoi(tempData[2]);
							if (backpropWeightsAuto != 0 && tempData.size() >= 4) backpropWeightsAuto->chosenIndex = stoi(tempData[3]);

							cout << "Loaded cached kernel values for layer " << tempData[0] << endl;
						}
					}
				}
			}
			loadConf.close();
		}
	}
#pragma endregion

	bool afterRestart = false;
	int restartEpoch = 0;
	int restartBatch = 0;
	float restartAnnealedLearningRate = 0;
	int restartNumRight = 0;
	float restartLoss = 0;
	if (config.loadWeights && config.weightsFile != "")
	{
		cout << "loadingweights" << endl;
		afterRestart = WeightsPersister::loadWeights(config.weightsFile, config.getTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss);
		if (!afterRestart && FileHelper::exists(config.weightsFile))
		{
			// try old trainingstring
			afterRestart = WeightsPersister::loadWeights(config.weightsFile, config.getOldTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss);
		}
		if (!afterRestart && FileHelper::exists(config.weightsFile))
		{
			cout << "Weights file " << config.weightsFile << " exists, but doesnt match training options provided." << endl;
			cout << "Continue loading anyway (might crash, or weights might be completely inappropriate)? (y/n)" << endl;
			string response;
			cin >> response;
			if (response != "y")
			{
				cout << "Please either check the training options, or choose a weights file that doesnt exist yet" << endl;
				return;
			}
		}
		cout << "reloaded epoch=" << restartEpoch << " batch=" << restartBatch << " numRight=" << restartNumRight << " loss=" << restartLoss << endl;
	}

	timer.timeCheck("before learning start");
	if (config.dumpTimings)
	{
		StatefulTimer::dump(true);
	}
	StatefulTimer::timeCheck("START");

	Trainable *trainable = net;
	MultiNet *multiNet = 0;
	if (config.multiNet > 1)
	{
		multiNet = new MultiNet(config.multiNet, net);
		trainable = multiNet;
	}
	NetLearnerBase *netLearner = 0;
	netLearner = new NetLearner(trainer, trainable,
		Ntrain, trainData, trainLabels,
		Ntest, testData, testLabels,
		config.batchSize
	);
	//    netLearner->setTrainer(trainer);
	netLearner->reset();
	netLearner->setSchedule(config.numEpochs, afterRestart ? restartEpoch : 0);
	if (afterRestart)
	{
		netLearner->setBatchState(restartBatch, restartNumRight, restartLoss);
	}
	netLearner->setDumpTimings(config.dumpTimings);
	//    netLearner->setLearningRate(config.learningRate, config.annealLearningRate);
	Timer weightsWriteTimer;

	while (!netLearner->isLearningDone())
	{
		//        netLearnerBase->tickEpoch();
		netLearner->tickBatch();
		if (netLearner->getEpochDone())
		{
			//            cout << "epoch done" << endl;
			if (config.weightsFile != "")
			{
				cout << "record epoch=" << netLearner->getNextEpoch() << endl;
				WeightsPersister::persistWeights(config.weightsFile.substr(0, config.weightsFile.size() - 4) + toString(netLearner->getNextEpoch()) + ".dat", config.getTrainingString(), net, netLearner->getNextEpoch(), 0, 0, 0, 0);
				weightsWriteTimer.lap();
			}
			//            Sampler::sampleFloatWrapper("conv weights", net->getLayer(6)->getWeightsWrapper());
			//            Sampler::sampleFloatWrapper("fc weights", net->getLayer(11)->getWeightsWrapper());
			//            Sampler::sampleFloatWrapper("conv bias", net->getLayer(6)->getBiasWrapper());
			//            Sampler::sampleFloatWrapper("fc bias", net->getLayer(11)->getBiasWrapper());
			if (config.dumpTimings)
			{
				StatefulTimer::dump(true);
			}

#pragma region Save kernel cache
			if (config.kernelCacheFile.length() > 0)
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
							cacheFile << "|-1";
						else
							cacheFile << "|" << toString(forwardAuto->chosenIndex);

						BackwardAuto *backwardAuto = dynamic_cast<BackwardAuto *>(conv->backwardImpl);

						if (backwardAuto == 0)
							cacheFile << "|-1";
						else
							cacheFile << "|" << toString(backwardAuto->chosenIndex);

						BackpropWeightsAuto *backpropWeightsAuto = dynamic_cast<BackpropWeightsAuto *>(conv->backpropWeightsImpl);

						if (backpropWeightsAuto == 0)
							cacheFile << "|-1" << endl;
						else
							cacheFile << "|" << toString(backpropWeightsAuto->chosenIndex) << endl;
					}
				}
				cacheFile.close();
			}
#pragma endregion

		}
		else
		{
			if (config.writeWeightsInterval > 0)
			{
				//                cout << "batch done" << endl;
				float timeMinutes = weightsWriteTimer.interval() / 1000.0f / 60.0f;
				//                cout << "timeMinutes " << timeMinutes << endl;
				if (timeMinutes >= config.writeWeightsInterval)
				{
					int nextEpoch = netLearner->getNextEpoch();
					int nextBatch = netLearner->getNextBatch();
					int batchNumRight = netLearner->getBatchNumRight();
					float batchLoss = netLearner->getBatchLoss();
					cout << "record epoch=" << nextEpoch << " batch=" << nextBatch <<
						"(" << ((float)nextBatch * 100.0f / netLearner->getNTrain() * config.batchSize) << "% of epoch)" <<
						" numRight=" << batchNumRight << "(" << (batchNumRight * 100.0f / nextBatch / config.batchSize) << "%)" <<
						" loss=" << batchLoss << endl;
					WeightsPersister::persistWeights(config.weightsFile, config.getTrainingString(), net,
						nextEpoch, nextBatch, 0, batchNumRight, batchLoss);
					weightsWriteTimer.lap();
				}
			}
		}
	}

	delete weightsInitializer;
	delete trainer;
	delete netLearner;
	if (multiNet != 0)
	{
		delete multiNet;
	}
	delete net;
	if (trainData != 0)
	{
		delete[] trainData;
	}
	if (testData != 0)
	{
		delete[] testData;
	}
	if (testLabels != 0)
	{
		delete[] testLabels;
	}
	if (trainLabels != 0)
	{
		delete[] trainLabels;
	}
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
		for (name,type,description,_, is_public_api) in options:
			if is_public_api:
				cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
		cog.outl('cout << "" << endl; ')
		cog.outl('cout << "unstable, might change within major version:" << endl; ')
		for (name,type,description,_, is_public_api) in options:
			if not is_public_api:
				cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
	*///]]]
	// generated using cog:
	cout << "public api, shouldnt change within major version:" << endl;
	cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
	cout << "    datadir=[directory to search for train and validate files] (" << config.dataDir << ")" << endl;
	cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
	cout << "    dataset=[choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]] (" << config.dataset << ")" << endl;
	cout << "    validatefile=[path to validation data file] (" << config.validateFile << ")" << endl;
	cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
	cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
	cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
	cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
	cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
	cout << "    loadweights=[load weights from file at startup?] (" << config.loadWeights << ")" << endl;
	cout << "    weightsfile=[file to write weights to] (" << config.weightsFile << ")" << endl;
	cout << "    writeweightsinterval=[write weights every this many minutes] (" << config.writeWeightsInterval << ")" << endl;
	cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
	cout << "    normalizationnumstds=[with stddev normalization, how many stddevs from mean is 1?] (" << config.normalizationNumStds << ")" << endl;
	cout << "    dumptimings=[dump detailed timings each epoch? [1|0]] (" << config.dumpTimings << ")" << endl;
	cout << "    multinet=[number of Mcdnn columns to train] (" << config.multiNet << ")" << endl;
	cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
	cout << "    filereadbatches=[how many batches to read from file each time? (for loadondemand=1)] (" << config.fileReadBatches << ")" << endl;
	cout << "    normalizationexamples=[number of examples to read to determine normalization parameters] (" << config.normalizationExamples << ")" << endl;
	cout << "    weightsinitializer=[initializer for weights, choices: original, uniform (default: original)] (" << config.weightsInitializer << ")" << endl;
	cout << "    trainer=[which trainer, sgd, anneal, nesterov, adagrad, rmsprop, or adadelta (default: sgd)] (" << config.trainer << ")" << endl;
	cout << "    learningrate=[learning rate, a float value, used by all trainers] (" << config.learningRate << ")" << endl;
	cout << "    momentum=[momentum, used by sgd and nesterov trainers] (" << config.momentum << ")" << endl;
	cout << "    weightdecay=[weight decay, 0 means no decay; 1 means full decay, used by sgd trainer] (" << config.weightDecay << ")" << endl;
	cout << "" << endl;
	cout << "unstable, might change within major version:" << endl;
	cout << "    initialweights=[for uniform initializer, weights will be initialized randomly within range -initialweights to +initialweights, divided by fanin, (default: 1.0f)] (" << config.initialWeights << ")" << endl;
	cout << "    rho=[rho decay, in adadelta trainer. 1 is no decay. 0 is full decay (default 0.9)] (" << config.rho << ")" << endl;
	cout << "    anneal=[multiply learningrate by this amount each epoch, used by anneal trainer, default 1.0] (" << config.anneal << ")" << endl;
	// [[[end]]]
}

int main(int argc, char *argv[])
{
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
							for (name,type,description,_,_) in options:
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
			else if (key == "datadir")
			{
				config.dataDir = (value);
			}
			else if (key == "trainfile")
			{
				config.trainFile = (value);
			}
			else if (key == "dataset")
			{
				config.dataset = (value);
			}
			else if (key == "validatefile")
			{
				config.validateFile = (value);
			}
			else if (key == "numtrain")
			{
				config.numTrain = atoi(value);
			}
			else if (key == "numtest")
			{
				config.numTest = atoi(value);
			}
			else if (key == "batchsize")
			{
				config.batchSize = atoi(value);
			}
			else if (key == "numepochs")
			{
				config.numEpochs = atoi(value);
			}
			else if (key == "netdef")
			{
				config.netDef = (value);
			}
			else if (key == "loadweights")
			{
				config.loadWeights = atoi(value);
			}
			else if (key == "weightsfile")
			{
				config.weightsFile = (value);
			}
			else if (key == "writeweightsinterval")
			{
				config.writeWeightsInterval = atof(value);
			}
			else if (key == "normalization")
			{
				config.normalization = (value);
			}
			else if (key == "normalizationnumstds")
			{
				config.normalizationNumStds = atof(value);
			}
			else if (key == "dumptimings")
			{
				config.dumpTimings = atoi(value);
			}
			else if (key == "multinet")
			{
				config.multiNet = atoi(value);
			}
			else if (key == "loadondemand")
			{
				config.loadOnDemand = atoi(value);
			}
			else if (key == "filereadbatches")
			{
				config.fileReadBatches = atoi(value);
			}
			else if (key == "normalizationexamples")
			{
				config.normalizationExamples = atoi(value);
			}
			else if (key == "weightsinitializer")
			{
				config.weightsInitializer = (value);
			}
			else if (key == "initialweights")
			{
				config.initialWeights = atof(value);
			}
			else if (key == "trainer")
			{
				config.trainer = (value);
			}
			else if (key == "learningrate")
			{
				config.learningRate = atof(value);
			}
			else if (key == "rho")
			{
				config.rho = atof(value);
			}
			else if (key == "momentum")
			{
				config.momentum = atof(value);
			}
			else if (key == "weightdecay")
			{
				config.weightDecay = atof(value);
			}
			else if (key == "anneal")
			{
				config.anneal = atof(value);
				// [[[end]]]
			}
			else if (key == "kernelCacheFile")
			{
				config.kernelCacheFile = (value);
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
	string dataset = toLower(config.dataset);
	if (dataset != "")
	{
		if (dataset == "mnist")
		{
			config.dataDir = "../data/mnist";
			config.trainFile = "train-images-idx3-ubyte";
			config.validateFile = "t10k-images-idx3-ubyte";
		}
		else if (dataset == "norb")
		{
			config.dataDir = "../data/norb";
			config.trainFile = "training-shuffled-dat.mat";
			config.validateFile = "testing-sampled-dat.mat";
		}
		else if (dataset == "cifar10")
		{
			config.dataDir = "../data/cifar10";
			config.trainFile = "train-dat.mat";
			config.validateFile = "test-dat.mat";
		}
		else if (dataset == "kgsgo")
		{
			config.dataDir = "../data/kgsgo";
			config.trainFile = "kgsgo-train10k-v2.dat";
			config.validateFile = "kgsgo-test-v2.dat";
			config.loadOnDemand = 1;
		}
		else if (dataset == "kgsgoall")
		{
			config.dataDir = "../data/kgsgo";
			config.trainFile = "kgsgo-trainall-v2.dat";
			config.validateFile = "kgsgo-test-v2.dat";
			config.loadOnDemand = 1;
		}
		else
		{
			cout << "dataset " << dataset << " not known.  please choose from: mnist, norb, cifar10, kgsgo" << endl;
			return -1;
		}
		cout << "Using dataset " << dataset << ":" << endl;
		cout << "   datadir: " << config.dataDir << ":" << endl;
		cout << "   trainfile: " << config.trainFile << ":" << endl;
		cout << "   validatefile: " << config.validateFile << ":" << endl;
	}
	try
	{
		go(config);
	}
	catch (runtime_error e)
	{
		cout << "Something went wrong: " << e.what() << endl;
		return -1;
	}
}


