#include "SimpleNeuron.h"

double SimpleNeuron::eta = 0.15;
double SimpleNeuron::alpha = 0.5;

SimpleNeuron::SimpleNeuron(size_t numOutputs, size_t index){
	m_index = index;
	for (size_t c = 0; c < numOutputs; ++c){
		m_weights.push_back(SimpleNeuronConnection());
	}
}

void SimpleNeuron::setOutputVal(double value){
	m_output = value;
}

double SimpleNeuron::getOutputVal(){
	return m_output;
}

double SimpleNeuron::transferFunction(double x){
	return tanh(x);
}

double SimpleNeuron::transferFunctionDerivative(double x){
	return 1.0 - x*x;
}

void SimpleNeuron::calcOutputGradients(double targetValue){
	double delta = targetValue - m_output;
	m_gradient = delta * SimpleNeuron::transferFunctionDerivative(m_output);
}

void SimpleNeuron::calcHiddenGradients(const SimpleNeuronLayer &nextLayer){
	double dow = sumDow(nextLayer);
	m_gradient = dow * SimpleNeuron::transferFunctionDerivative(m_output);
}

double SimpleNeuron::sumDow(const SimpleNeuronLayer &nextLayer) const{
	double sum = 0.0;
	for (size_t n = 0; n<nextLayer.size() - 1; ++n){
		sum += m_weights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void SimpleNeuron::updateInputWeights(SimpleNeuronLayer &prevLayer){
	for (auto &neuron : prevLayer){
		double oldDeltaWeight = neuron.m_weights[m_index].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
		neuron.m_weights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_weights[m_index].weight += newDeltaWeight;
	}
}

void SimpleNeuron::feedForward(const SimpleNeuronLayer &prevLayer){
	double sum = 0.0;

	for (auto &neuron : prevLayer){
		sum += neuron.m_output * neuron.m_weights[m_index].weight;
	}

	m_output = SimpleNeuron::transferFunction(sum);
}