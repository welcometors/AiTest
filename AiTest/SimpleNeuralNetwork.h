#pragma once
#ifndef __SIMPLE_NEURAL_NETWORK_H__
#define __SIMPLE_NEURAL_NETWORK_H__

#include "SimpleNeuron.h"

class SimpleNeuralNetwork{
public:
	SimpleNeuralNetwork(const std::vector<size_t> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backPropogation(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &outputVals);
	double getRunningAverage();
	static double const m_runningAverageSmoothFactor;

private:
	std::vector<SimpleNeuronLayer> m_layers;
	double m_error;
	double m_runningAverage;
};

#endif