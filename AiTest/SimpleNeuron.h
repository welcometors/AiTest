#ifndef __SIMPLE_NEURON_H__
#define __SIMPLE_NEURON_H__

#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct SimpleNeuronConnection{
	double weight;
	double deltaWeight;

	SimpleNeuronConnection(){
		weight = rand() / double(RAND_MAX);
	}
};

class SimpleNeuron;
typedef vector<SimpleNeuron> SimpleNeuronLayer;

class SimpleNeuron{
public:
	SimpleNeuron(size_t numOutputs, size_t index);
	void setOutputVal(double value);
	double getOutputVal();
	void feedForward(const SimpleNeuronLayer &prevLayer);
	void calcOutputGradients(double targetValue);
	void calcHiddenGradients(const SimpleNeuronLayer &nextLayer);
	double sumDow(const SimpleNeuronLayer &nextLayer) const;
	void updateInputWeights(SimpleNeuronLayer &prevLayer);
private:
	static double eta;
	static double alpha;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double m_output;
	double m_gradient;
	size_t m_index;
	vector<SimpleNeuronConnection> m_weights;
};

#endif