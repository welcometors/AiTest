#ifndef __SIMPLE_NEURAL_NETWORK_H__
#define __SIMPLE_NEURAL_NETWORK_H__

#include "SimpleNeuron.h"

class SimpleNeuralNetwork{
public:
	SimpleNeuralNetwork(const vector<int> &topology);
	void feedForward(const vector<double> &inputVals);
	void backPropogation(const vector<double> &targetVals);
	void getResults(vector<double> &outputVals);

private:
	vector<SimpleNeuronLayer> m_layers;
	double m_error;
	double m_runningAverage;
	static double const m_runningAverageSmoothFactor;
};

#endif