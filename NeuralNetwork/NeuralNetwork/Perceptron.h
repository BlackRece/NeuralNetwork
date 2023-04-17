#pragma once
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include "Activation.h"

class Perceptron
{
public:
	Perceptron();
	~Perceptron();

	void SetInputs(std::vector<float> vecInputs) { m_vecInputs = vecInputs; }
	void SetWeights(std::vector<float> vecWeights) { m_vecWeights = vecWeights; }

	int GetOutput() 
	{
		if (m_vecInputs.size() != m_vecWeights.size())
			return 0;

		float fSum = 0;
		for (int i = 0; i < m_vecWeights.size(); i++)
		{
			fSum += m_vecInputs[i] * m_vecWeights[i];
		}

		return Activation::Sign(fSum);
	}
private:
	std::vector<float> m_vecInputs;
	std::vector<float> m_vecWeights;
};

Perceptron::Perceptron()
{
}

Perceptron::~Perceptron()
{
}
#endif // !PERCEPTRON_H

