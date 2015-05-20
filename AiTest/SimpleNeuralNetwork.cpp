#include "SimpleNeuralNetwork.h"

SimpleNeuralNetwork::SimpleNeuralNetwork(const vector<int> &topology){
	size_t numLayers = topology.size();

	for (size_t i = 0; i < numLayers; ++i){
		m_layers.push_back(SimpleNeuronLayer());
		size_t numOutput = (i == numLayers - 1) ? 0 : topology[i + 1];
		for (unsigned j = 0; j <= topology[i]; ++j){
			m_layers.back().push_back(SimpleNeuron(numOutput, j));
		}
		m_layers.back().back().setOutputVal(1.0);
	}
}

void SimpleNeuralNetwork::feedForward(const vector<double> &inputVals){
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (size_t i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	for (size_t i = 1; i < m_layers.size(); ++i){
		for (size_t j = 0; j < m_layers[i].size() - 1; ++j){
			m_layers[i][j].feedForward(m_layers[i - 1]);
		}
	}
}

void SimpleNeuralNetwork::backPropogation(const vector<double> &targetVals){
	// Calculate overall net error
	SimpleNeuronLayer  &outputLayer = m_layers.back();
	m_error = 0.0;
	for (size_t n = 0; n < outputLayer.size() - 1; ++n){
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta*delta;
	}
	m_error = sqrt(m_error / (outputLayer.size() - 1));

	m_runningAverage =
		(m_runningAverage * m_runningAverageSmoothFactor + m_error)
		/ (m_runningAverageSmoothFactor + 1.0);

	// Calculate output layer gradients
	for (size_t n = 0; n < outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers
	for (size_t layerNum = m_layers.size() - 2; layerNum >0; --layerNum){
		SimpleNeuronLayer &hiddenLayer = m_layers[layerNum];
		SimpleNeuronLayer &nextLayer = m_layers[layerNum + 1];

		for (size_t n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// Update weights for all layers
	for (size_t layerNum = m_layers.size() - 1; layerNum >0; --layerNum){
		SimpleNeuronLayer &layer = m_layers[layerNum];
		SimpleNeuronLayer &nextLayer = m_layers[layerNum + 1];

		for (size_t n = 0; n < layer.size(); ++n){
			layer[n].updateInputWeights(nextLayer);
		}
	}
}

void SimpleNeuralNetwork::getResults(vector<double> &outputVals){
	outputVals.clear();
	SimpleNeuronLayer &outputLayer = m_layers.back();

	for (size_t n = 0; n < outputLayer.size() - 1; ++n){
		outputVals.push_back(outputLayer[n].getOutputVal());
	}
}