#include "NN.h"

NN::NN(int iInputCount, int iHiddenCount, int iOutputCount)
	: m_iInputCount(iInputCount), m_iHiddenCount(iHiddenCount), m_iOutputCount(iOutputCount)
{
	// prepare function to random every element in matrix
	std::function<double()> fnRandomDouble = std::bind(&NN::random, this, -1.0, 1.0);

	// create matrices
	m_mInput = Matrix(m_iInputCount, 1);
	m_mHidden = Matrix(m_iHiddenCount, 1);
	m_mOutput = Matrix(m_iOutputCount, 1);

	m_mWeightsIH = Matrix(m_iHiddenCount, m_iInputCount);
	m_mWeightsHO = Matrix(m_iOutputCount, m_iHiddenCount);

	m_mBiasH = Matrix(m_iHiddenCount, 1);
	m_mBiasO = Matrix(m_iOutputCount, 1);

	// randomize weights and biases
	m_mWeightsIH = m_mWeightsIH.map(fnRandomDouble);
	m_mWeightsHO = m_mWeightsHO.map(fnRandomDouble);

	m_mBiasH = m_mBiasH.map(fnRandomDouble);
	m_mBiasO = m_mBiasO.map(fnRandomDouble);

	// set default learning rate
	m_dLearningRate = LEARNING_RATE;
}

double* NN::feedForward(const double dInputs[], const int iInputCount)
{
	// convert input array to matrix
	bool bIsColumnVector = true;
	m_mInput = m_mInput.fromArray(dInputs, iInputCount, bIsColumnVector);

	// apply weights and biases to input
	m_mHidden = m_mWeightsIH.dot(m_mInput);
	m_mHidden = m_mHidden.add(m_mBiasH);

	// prepare and apply activation function
	std::function<double(double)> fnSigmoid = std::bind(&NN::sigmoid, this, std::placeholders::_1);
	m_mHidden = m_mHidden.map(fnSigmoid);

	// apply weights and biases to hidden layer
	m_mOutput = m_mWeightsHO.dot(m_mHidden);
	m_mOutput = m_mOutput.add(m_mBiasO);

	// apply activation function
	m_mOutput = m_mOutput.map(fnSigmoid);

	// return result as array
	return m_mOutput.toArray();
}

void NN::trainFeedForward(const double dInputs[], const int iInputCount, const double dTargets[], const int iTargetCount)
{
	double* rawOutputs = feedForward(dInputs, iInputCount);

	Matrix mOutputs, mTargets;
	// get output matrix from feed forward
	mOutputs = mOutputs.fromArray(rawOutputs, m_iOutputCount);

	// convert target array to matrix
	mTargets = mTargets.fromArray(dTargets, iTargetCount);

	// calculate output errors
	Matrix mOutputErrors = mTargets.sub(mOutputs);

	// calculate hidden-output gradients via gradient descent
	std::function<double(double)> fnSigmoidDerivative = std::bind(&NN::sigmoidDerivative, this, std::placeholders::_1);
	Matrix mOutputGradients = mOutputs.map(fnSigmoidDerivative);
	mOutputGradients = mOutputGradients.mul(mOutputErrors);

	double dLearningRate = 1.0;		// TODO: make this a member variable that starts at 0.1 and decreases over time
	mOutputGradients = mOutputGradients.mul(dLearningRate);

	// calculate hidden-output deltas
	Matrix mHidden_T = m_mHidden.transpose();
	Matrix mWeightsHO_deltas = mOutputGradients.dot(mHidden_T);

	// adjust hidden-output weights and biases
	m_mWeightsHO = m_mWeightsHO.add(mWeightsHO_deltas);
	
	// apply hidden->output biases
	m_mBiasO = m_mBiasO.add(mOutputGradients);

	// calculate hidden->output deltas/errors
	Matrix mWeightsHO_T = m_mWeightsHO.transpose();
	Matrix mHiddenErrors = mWeightsHO_T.dot(mOutputErrors);

	// calculate input->hidden gradients via gradient descent
	Matrix mHiddenGradients = m_mHidden.map(fnSigmoidDerivative);
	mHiddenGradients = mHiddenGradients.mul(mHiddenErrors);
	mHiddenGradients = mHiddenGradients.mul(dLearningRate);
	
	// calculate input->hidden deltas
	Matrix mInput_T = m_mInput.transpose();
	Matrix mWeightsIH_deltas = mHiddenGradients.dot(mInput_T);

	// adjust input->hidden weights and biases
	m_mWeightsIH = m_mWeightsIH.add(mWeightsIH_deltas);
	
	// apply input->hidden biases
	m_mBiasH = m_mBiasH.add(mHiddenGradients);

}

void NN::save(const std::string sFileName)
{
	// save json file using JsonParser
	NNJson json;
	json.dLearningRate = m_dLearningRate;

	json.iInputNodes = m_iInputCount;
	json.iHiddenNodes = m_iHiddenCount;
	json.iOutputNodes = m_iOutputCount;

	json.mInputWeights = m_mWeightsIH.toJson();
	json.mHiddenWeights = m_mWeightsHO.toJson();

	json.mBiasHidden = m_mBiasH.toJson();
	json.mBiasOutput = m_mBiasO.toJson();

	JsonParser::SaveJson(json, sFileName);
}

void NN::load(const std::string sFileName)
{
	// load json file using JsonParser
	NNJson json;
	JsonParser::LoadJson(json, sFileName);
	if (json.iInputNodes == 0 || json.iHiddenNodes == 0 || json.iOutputNodes == 0)
	{
		std::cout << "Error: Invalid JSON file" << std::endl;
		return;
	}

	m_dLearningRate = json.dLearningRate;

	m_iInputCount = json.iInputNodes;
	m_iHiddenCount = json.iHiddenNodes;
	m_iOutputCount = json.iOutputNodes;

	m_mWeightsIH = m_mWeightsIH.fromJson(json.mInputWeights);
	m_mWeightsHO = m_mWeightsHO.fromJson(json.mHiddenWeights);

	m_mBiasH = m_mBiasH.fromJson(json.mBiasHidden);
	m_mBiasO = m_mBiasO.fromJson(json.mBiasOutput);
}
