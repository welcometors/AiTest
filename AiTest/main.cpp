#include <iostream>
#include <vector>
#include <cmath>
#include "SimpleNeuralNetwork.h"

#define DEBUG
const int numOfTrainningSamples = 10000;
const int numOfTestSamples = numOfTrainningSamples * .15;

template<size_t I, size_t O>
struct TraningData{
	std::vector<double> inputVals;
	std::vector<double> outputVals;
	std::vector<double> actualVals;

	TraningData(){
		double ang = (1 + (rand() % 78)) / 100.0;
		//if (rand() & 1) ang *= -1;
		inputVals.push_back(cos(ang));
		inputVals.push_back(sin(ang));
		actualVals.push_back(ang);
		outputVals.push_back(0);
	}

	void print(double ra){
#ifdef DEBUG
		//printf("Training: %d ^ %d = %d, out: %f, RA: %f\n", a, b, c, outputVals[0], snn.getRunningAverage());
		printf("Training: tan-1(%f, %f) = %f, out: %f, RA: %f\n", inputVals[0], inputVals[1], actualVals[0], outputVals[0], ra);
#endif
	}
};

int main(){
	SimpleNeuralNetwork snn({ 2, 4, 1 });
	
	for (int i = 0; i < numOfTrainningSamples; i++){
		TraningData<2, 1> trainingData;
		snn.feedForward(trainingData.inputVals);
		snn.getResults(trainingData.outputVals);
		snn.backPropogation(trainingData.actualVals);
		//trainingData.print(snn.getRunningAverage());
	}
	printf("Training done, running average: %f\n", snn.getRunningAverage());

	double rms = 0.0;
	for (int i = 0; i < numOfTestSamples; i++){
		TraningData<2, 1> trainingData;
		snn.feedForward(trainingData.inputVals);
		snn.getResults(trainingData.outputVals);
		trainingData.print(snn.getRunningAverage());
		rms += (trainingData.actualVals[0] - trainingData.outputVals[0])*(trainingData.actualVals[0] - trainingData.outputVals[0]);
	}

	TraningData<2, 1> trainingData;
	trainingData.inputVals[0] = 0.93937271284737892003503235730367;
	trainingData.inputVals[1] = 0.34289780745545134918963490691763;
	trainingData.actualVals[0] = 0.35;
	snn.feedForward(trainingData.inputVals);
	snn.getResults(trainingData.outputVals);
	trainingData.print(snn.getRunningAverage());

	printf("Average error is (%3.5f%%) after traing %d sample data.\n", sqrt(rms / numOfTestSamples), numOfTrainningSamples);
	system("pause");
} 