#pragma once
#ifndef ACTIVATION_H
#define ACTIVATION_H

class Activation
{
public:
	static int Sign(float fSum) { return fSum > 0 ? 1 : -1; }

private:
	Activation();
	~Activation();

};

Activation::Activation()
{
}

Activation::~Activation()
{
}
#endif // !ACTIVATION_H
