#pragma once
#include "Neuron.h"
#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork() ;
	NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize);
	~NeuralNetwork();

	inline static double InitWeights() { return RND::Getf(); }
	
	inline static double Sigmoid(double x) { return 1 / (1 + exp(-x)); }
	inline static double SigmoidDerivative(double x) { return x * (1 - x); }
	
	static void Shuffle(std::vector<int>& vInput);

private:
	int iInputLayerSize;
	int iHiddenLayerSize;
	int iOutputLayerSize;
	
	std::shared_ptr<Neuron> pNeuron;
	 
};

