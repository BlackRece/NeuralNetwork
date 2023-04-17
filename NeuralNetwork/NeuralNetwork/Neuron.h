#pragma once
#ifndef NEURON_H
#define NEURON_H

#include "Random.h"

class Neuron
{
public:
	Neuron() : m_fBias(RND::Getf()), m_fWeight(RND::Getf()) {}
	~Neuron() {}
	
	inline void SetBias(const float fBias) noexcept { m_fBias = fBias; }
	inline void SetWeight(const float fWeight) noexcept { m_fWeight = fWeight; }

private:
	float m_fBias;
	float m_fWeight;
};

#endif // !NEURON_H
