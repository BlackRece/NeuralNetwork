// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <functional>
#include "NeuralNetwork.h"
#include "Perceptron.h"
#include "Matrix.h"
#include "NN.h"

#define numInputs 2
#define numHiddenNodes 4
#define numOutputs 1
#define numTrainingSets 4

#define numEpochIterations 50000

void testNNConstructor()
{
	std::cout << "\n\nNeural Network Test for Constructor\n";

	double dInputs[numTrainingSets][numInputs] = {
		{0,0},
		{0,1},
		{1,0},
		{1,1}
	};
	double dTargets[numTrainingSets][numOutputs] = {
		{0},
		{1},
		{1},
		{0}
	};

	NN nn1 = NN(numInputs, numHiddenNodes, numOutputs);
	NN nn2 = nn1.clone();

	std::cout << "\n\nnn1 output\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn1.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}

	std::cout << "\n\nnn2 output\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn2.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}

	std::cout << "\n\ntrain nn2\n";
	for (int i = 0; i < numEpochIterations; i++)
	{
		int index = rand() % numTrainingSets;
		nn2.trainFeedForward(dInputs[index], numInputs, dTargets[index], numOutputs);
	}

	std::cout << "\n\nnn1 output\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn1.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}

	std::cout << "\n\nnn2 output\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn2.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}

	std::cout << "\n\nmutate nn2\n";
	nn2.mutate(0.1);

	std::cout << "\n\nnn2's mutated output\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn2.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}
}

void testNNJsonOperations()
{
	std::cout << "\n\nNeural Network Test for Json Operations\n";
	NN nn1 = NN(numInputs, numHiddenNodes, numOutputs);
	nn1.setLearningRate(0.5);
	double dInputs[numTrainingSets][numInputs] = {
		{0,0},
		{0,1},
		{1,0},
		{1,1}
	};
	double dTargets[numTrainingSets][numOutputs] = {
		{0},
		{1},
		{1},
		{0}
	};

	for (int i = 0; i < numEpochIterations; i++)
	{
		int index = rand() % numTrainingSets;
		nn1.trainFeedForward(dInputs[index], numInputs, dTargets[index], numOutputs);
	}

	std::cout << "\n\n get output matrix from feed forward\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn1.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}

	std::cout << "\n\n save trained nn\n";
	nn1.save("nnTest.json");

	NN nn2 = NN(99,99,99);
	std::cout << "\n\n load nn\n";
	nn2.load("nnTest.json");
	std::cout << "\n\n get stored learning rate: " << nn2.getLearningRate() << "\n";
	std::cout << "\n\n get output matrix from feed forward of stored nn\n";

	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn2.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}
}

void testTrainingForXOR()
{
	// supervised learning
	std::cout << "\n\nNeural Network Test for training XOR\n";
	NN nn = NN(numInputs, numHiddenNodes, numOutputs);

	double dInputs[numTrainingSets][numInputs] = { 
		{0,0},
		{0,1},
		{1,0},
		{1,1} 
	};
	double dTargets[numTrainingSets][numOutputs] = {
		{0},
		{1},
		{1},
		{0}
	};

	for (int i = 0; i < numEpochIterations; i++)
	{
		int index = rand() % numTrainingSets;
		nn.trainFeedForward(dInputs[index], numInputs, dTargets[index], numOutputs);
	}

	std::cout << "\n\n get output matrix from feed forward\n";
	for (int i = 0; i < numTrainingSets; i++)
	{
		double* rawOutput = nn.feedForward(dInputs[i], numInputs);
		Matrix* mOutputs{};
		mOutputs = mOutputs->fromArray(rawOutput, numOutputs);
		std::cout << "\n" << i << ": (target = " << dTargets[0][i] << ")\n";
		mOutputs->print();
	}
}

void testTrainFeedForward()
{
	//std::cout << "\n\nNeural Network Test for training from Feed Forward method\n";
	//NN nn = NN(numInputs, numHiddenNodes, numOutputs);
	//std::cout << "\n nn created with 2 inputs, 2 hidden nodes, 2 outputs\n";

	//double dInputs[numInputs] = { 1, 0 };
	//std::cout << "\n setting inputs to { 1, 0 }\n";

	//double dTargets[numOutputs] = { 1 };
	//std::cout << "\n setting targets to { 1 }\n";

	//double* rawOutput = nn.feedForward(dInputs, numInputs);
	//std::cout << "\n raw output from feedForward:\n";
	//for (int i = 0; i < numOutputs; i++)
	//	std::cout << i << ": " << rawOutput[i] << std::endl;

	//Matrix mOutputs, mTargets;
	//std::cout << "\n\n get output matrix from feed forward\n";
	//mOutputs = mOutputs.fromArray(rawOutput, numOutputs);
	//mOutputs.print();

	//std::cout << "\n convert target array to matrix\n";
	//mTargets = mTargets.fromArray(dTargets, numOutputs);
	//mTargets.print();

	//std::cout << "\n calculate output errors\n";
	//Matrix mOutputErrors = mTargets.sub(mOutputs);
	//mOutputErrors.print();

	//std::cout << "\n pre-transposed weights:\n";
	//nn.m_mWeightsHO.print();
	//Matrix mWeightsHO_T = nn.m_mWeightsHO.transpose();
	//std::cout << "\n post-transposed weights:\n";
	//mWeightsHO_T.print();

	//std::cout << "\n calculate hidden layer errors\n";
	//Matrix mHiddenErrors = mWeightsHO_T.dot(mOutputErrors);
	//mHiddenErrors.print();

	//std::cout << "\n apply sigmoid derivative to mOutputs\n";
	//std::function<double(double)> fnSigmoidDerivative = std::bind(&NN::sigmoidDerivative, &nn, std::placeholders::_1);
	//Matrix mGradients = mOutputs.map(fnSigmoidDerivative);
	//mGradients.print();

	//std::cout << "\n calculate gradients with learning rate\n";
	//mGradients = mGradients.mul(mOutputErrors);

	//double m_dLearningRate = 1.0;		// TODO: make this a member variable that starts at 0.1 and decreases over time
	//mGradients = mGradients.mul(m_dLearningRate);
	//mGradients.print();

	//std::cout << "\n get hidden matrix with weights applied\n";
	//nn.m_mHidden.print();

	//std::cout << "\n apply biases to hidden matrix\n";
	//Matrix mHiddenWithBias = nn.m_mHidden.add(nn.m_mBiasH);
	//mHiddenWithBias.print();

	//std::cout << "\n transpose hidden matrix with weights and biases alread applied\n";
	//Matrix mHiddenTransposed = mHiddenWithBias.transpose();
	//mHiddenTransposed.print();

	//std::cout << "\n calculate hidden -> output weight deltas\n";
	//Matrix mWeightsHO_deltas = mGradients.dot(mHiddenTransposed);
	//mWeightsHO_deltas.print();

	//std::cout << "\n apply hidden -> output weight deltas to previous hidden -> output weights";
	//std::cout << "\nnn.m_mWeightsHO.print();\n";
	//nn.m_mWeightsHO.print();
	//std::cout << "\nmAdjustedWeightsHO.print();\n";
	//Matrix mAdjustedWeightsHO = nn.m_mWeightsHO.add(mWeightsHO_deltas);
	//mAdjustedWeightsHO.print();

	//std::cout << "\n apply sigmoid derivative to mOutputs\n";
	//Matrix mHiddenGradients = mHiddenWithBias
}

void testFeedForward()
{
	std::cout << "\n\nNeural Network Test\n";
	NN nn = NN(numInputs, numHiddenNodes, numOutputs);
	std::cout << "\n nn created with 2 inputs, 2 hidden nodes, 2 outputs\n";

	double dInputs[numInputs] = { 1, 0 };
	std::cout << "\n setting inputs to { 1, 0 }\n";

	double* dOutputs = nn.feedForward(dInputs, numInputs);
	std::cout << "\n\nfeedForward({ 1, 0 })\n";
	for (int i = 0; i < numOutputs; i++)
	{
		std::cout << dOutputs[i] << std::endl;
	}
	dInputs[0] = 0;
	dInputs[1] = 1;
	dOutputs = nn.feedForward(dInputs, numInputs);
	std::cout << "\n\nfeedForward({ 0, 1 })\n";
	for (int i = 0; i < numOutputs; i++)
	{
		std::cout << dOutputs[i] << std::endl;
	}
	dInputs[0] = 1;
	dInputs[1] = 1;
	dOutputs = nn.feedForward(dInputs, numInputs);
	std::cout << "\n\nfeedForward({ 1, 1 })\n";
	for (int i = 0; i < numOutputs; i++)
	{
		std::cout << dOutputs[i] << std::endl;
	}
	dInputs[0] = 0;
	dInputs[1] = 0;
	dOutputs = nn.feedForward(dInputs, numInputs);
	std::cout << "\n\nfeedForward({ 0, 0 })\n";
	for (int i = 0; i < numOutputs; i++)
	{
		std::cout << dOutputs[i] << std::endl;
	}
}

void testMatrix()
{
	std::cout << "\ntesting random number generator:" << rand() % RAND_MAX << std::endl;
	std::cout << "\n\nMatrix Test";

	std::cout << "\npMatrix1\n";
	auto* pMatrix1 = new Matrix(3, 2);
	pMatrix1->print();

	std::cout << "\npMatrix2\n";
	auto* pMatrix2 = new Matrix(2, 3);
	pMatrix2->print();

	std::cout << "\n\nRandom Matrix Test\n";

	pMatrix1->random(100);
	std::cout << "\npMatrix1.random(100)\n";
	pMatrix1->print();

	pMatrix1->mul(0.10);
	std::cout << "\npMatrix1.mul(0.10)\n";
	pMatrix1->print();

	NN nn = NN(2, 2, 2);
	std::function<double()> fnRandomDouble = std::bind(&NN::random, &nn, -2.0, 2.0);
	std::cout << "\npMatrix1.map(fnRandomDouble);\n";
	pMatrix1->print();
	pMatrix1 = pMatrix1->map(fnRandomDouble);
	std::cout << "\npMatrix1 = pMatrix1.map(fnRandomDouble);\n";
	pMatrix1->print();

	std::cout << "\n\nMatrix Addition Test\n";

	Matrix* mMatrix3 = Matrix::add(*pMatrix1, *pMatrix2);
	std::cout << "\npMatrix1.add(pMatrix2)\nMatrix3 = \n";
	mMatrix3->print();

	std::cout << "\n\nMatrix Multiplication Test\n";

	auto* mMatrix4 = Matrix::add(*pMatrix2, 2);
	std::cout << "\npMatrix2.add(2)\nmMatrix4 = \n";
	mMatrix4->print();

	auto* mMatrix5 = Matrix::mul(*pMatrix1, *mMatrix4);
	std::cout << "\npMatrix1.mul(mMatrix4)\nmMatrix5 = \n";
	mMatrix5->print();

	std::cout << "\n\nMatrix Dot Product Test\n";

	std::cout << "\npMatrix1\n";
	pMatrix1->print();
	std::cout << std::endl;
	pMatrix1->showRow(1);
	std::cout << std::endl;
	std::cout << "\nmMatrix6\n";
	auto* mMatrix6 = new Matrix(2, 3);
	mMatrix6->print();
	std::cout << "\nmMatrix6.random()\n";
	mMatrix6->random();
	mMatrix6->print();
	std::cout << std::endl;
	mMatrix6->showCol(1);
	std::cout << std::endl;

	auto* mMatrix7 = Matrix::dot(*pMatrix1, *mMatrix6);
	std::cout << "\npMatrix1.dot(mMatrix6)\nmMatrix7 = \n";
	mMatrix7->print();

	std::cin;
	/*
	const double dLearningRate = 0.1f;
	
	double dHiddenLayer[numHiddenNodes];
	double dOutputLayer[numOutputs];

	double dHiddenBias[numHiddenNodes];
	double dOutputBias[numOutputs];
	
	double dHiddenWeights[numInputs][numHiddenNodes];
	double dOutputWeights[numHiddenNodes][numOutputs];
	
	double dTrainingInputs[numTrainingSets][numInputs] = {
		{0.0f, 0.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f},
		{1.0f, 1.0f}
	};

	double dTrainingOutputs[numTrainingSets][numOutputs] = {
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f}
	};
	
	// Initialize the weights and biases with random values
	for (int i = 0; i < numInputs; i++)
	{
		for (int j = 0; j < numHiddenNodes; j++)
		{
			dHiddenWeights[i][j] = NeuralNetwork::InitWeights();
		}
	}
	
	for (int i = 0; i < numHiddenNodes; i++)
	{
		for (int j = 0; j < numOutputs; j++)
		{
			dOutputWeights[i][j] = NeuralNetwork::InitWeights();
		}
	}

	for(double dOutput : dOutputBias)
		dOutput = NeuralNetwork::InitWeights();
	
	std::vector<int> vecTrainingSetOrder = { 0,1,2,3 };

	int numOfEpochs = 100;

	// Train the neural network for a number of epochs
	for (int iEpoch = 0; iEpoch < numOfEpochs; iEpoch++)
	{
		NeuralNetwork::Shuffle(vecTrainingSetOrder);

		for (int iOrderIndex = 0; iOrderIndex < vecTrainingSetOrder.size(); iOrderIndex++)
		{
			int iSetIndex = vecTrainingSetOrder[iOrderIndex];

			// Forward Pass

			// Compute hidden layer activation
			for (int iNodeIndex = 0; iNodeIndex < numHiddenNodes; iNodeIndex++)
			{
				double activation = dHiddenBias[iNodeIndex];

				for (int iInputIndex = 0; iInputIndex < numInputs; iInputIndex++)
				{
					activation += dTrainingInputs[iSetIndex][iInputIndex] * dHiddenWeights[iInputIndex][iNodeIndex];
				}

				dHiddenLayer[iNodeIndex] = NeuralNetwork::Sigmoid(activation);
			}

			// Compute output layer activation
			for (int iOutputIndex = 0; iOutputIndex < numOutputs; iOutputIndex++)
			{
				double activation = dOutputBias[iOutputIndex];

				for (int iNodeIndex = 0; iNodeIndex < numHiddenNodes; iNodeIndex++)
				{
					activation += dHiddenLayer[iNodeIndex] * dOutputWeights[iNodeIndex][iOutputIndex];
				}

				dOutputLayer[iOutputIndex] = NeuralNetwork::Sigmoid(activation);
			}

			std::cout
				<< "Input :" << dTrainingInputs[iSetIndex][0]
				<< "Output :" << dOutputLayer[0]
				<< "Predicted Output :" << dTrainingOutputs[iSetIndex][0]
				<< std::endl;

			// Back-propergation

			// Compute change in output weights
			double dDeltaOutputs[numOutputs];

			for (int j = 0; j < numOutputs; j++)
			{
				double dError = (dTrainingOutputs[iSetIndex][j] - dOutputLayer[j]);
				dDeltaOutputs[j] = dError * NeuralNetwork::SigmoidDerivative(dOutputLayer[j]);
			}

			// Compute change in hidden weights
			double dDeltaHidden[numHiddenNodes];

			for (int j = 0; j < numHiddenNodes; j++)
			{
				double dError = 0.0f;
				for (int k = 0; k < numOutputs; k++)
				{
					dError += dDeltaOutputs[k] * dOutputWeights[j][k];
				}
				dDeltaHidden[j] = dError * NeuralNetwork::SigmoidDerivative(dHiddenLayer[j]);
			}

			// Apply change in output weights
			for (int j = 0; j < numOutputs; j++)
			{
				dOutputBias[j] += dDeltaOutputs[j] * dLearningRate;
				for (int k = 0; k < numHiddenNodes; k++)
				{
					dOutputWeights[k][j] += dHiddenLayer[k] * dDeltaOutputs[j] * dLearningRate;
				}
			}

			// Apply change in hidden weights
			for (int j = 0; j < numHiddenNodes; j++)
			{
				dHiddenBias[j] += dDeltaHidden[j] * dLearningRate;
				for (int k = 0; k < numInputs; k++)
				{
					dHiddenWeights[k][j] += dTrainingInputs[iSetIndex][k] * dDeltaHidden[j] * dLearningRate;
				}
			}

			// Output Results
			std::cout << "Hidden Biases: \n";
			for (double dHBias : dHiddenBias)
				std::cout << dHBias << std::endl;

			std::cout << "Hidden Weights: \n";
			for (int j = 0; j < numOutputs; j++)
			{
				std::cout << "[ \n";
				for (int k = 0; k < numHiddenNodes; k++)
					std::cout<< dHiddenWeights[j][k] << std::endl;

				std::cout << "] \n";
			}

			std::cout << "Output Biases: \n";
			for (double dOBias : dOutputBias)
				std::cout << dOBias << std::endl;

			std::cout << "Output Weights: \n";
			for (int j = 0; j < numHiddenNodes; j++)
			{
				std::cout << "[ \n";
				for (int k = 0; k < numOutputs; k++)
					std::cout << dOutputWeights[j][k] << std::endl;

				std::cout << "] \n";
			}
		}
	}
	
	// Instantiate the neural network
	std::unique_ptr< NeuralNetwork > neuralNetwork = std::make_unique< NeuralNetwork >(2, 1, 1);
    */
}

void rawthoughts()
{
	std::cout << "Hello World!\n";

	// Populate the input layer of a neural network with random input data
	// Create a neural network with 3 layers
	// 1 input layer, 1 hidden layer, 1 output layer
	// 2 nodes in the input layer, 1 nodes in the hidden layer, 1 node in the output layer
	// Use a single artificial neuron in the hidden layer

	for (int iEpoch = 0; iEpoch < numEpochIterations; iEpoch++)
	{
		std::vector<float> vecInputs;
		for (int i = 0; i < numInputs; i++)
			vecInputs.push_back(RND::Getf(-5.0f, 5.0f));

		std::vector<float> vecWeights;
		for (int i = 0; i < numInputs; i++)
			vecWeights.push_back(RND::Getf(-5.0f, 5.0f));

		std::unique_ptr<Perceptron> pPerceptron = std::make_unique<Perceptron>();
		pPerceptron->SetInputs(vecInputs);
		pPerceptron->SetWeights(vecWeights);
		int iGuess = pPerceptron->GetOutput();

		std::cout
			<< "\nInputs: ";
		for (float fInput : vecInputs)
			std::cout << fInput << ", ";

		std::cout
			<< "\nWeights: ";
		for (float fWeight : vecWeights)
			std::cout << fWeight << ", ";

		std::cout
			<< "\nGuess: "
			<< iGuess
			<< std::endl;
	}

	std::cout
		<< "\nPress ENTER to continue...";

	std::cin;

}

int main()
{
	//testMatrix();

	//testFeedForward();

	//testTrainFeedForward();

	//testTrainingForXOR();

	//testNNJsonOperations();

	testNNConstructor();

	return 0;
}


NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
{
	
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::Shuffle(std::vector<int>& vInput)
{
	size_t iInputCount = vInput.size();
	if (iInputCount < 1)
		return;
	
	iInputCount--;

	for (size_t iIndex = 0; iIndex < iInputCount - 1; iIndex++)
	{
		size_t iRandomIndex = iIndex + RND::Get(iInputCount);
		int iTemp = vInput[iRandomIndex];
		vInput[iRandomIndex] = vInput[iIndex];
		vInput[iIndex] = iTemp;
	}

}