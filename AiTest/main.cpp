#include <iostream>
#include <vector>
#include <cmath>
#include "SimpleNeuralNetwork.h"

const int numOfTrainningSamples = 2000;
const int numOfTestSamples = 10;

int main(){
	SimpleNeuralNetwork snn({ 2, 4, 1 });

	std::vector<double> inputVals(2), outputVals, actualVals(1);
	for (int i = 0; i < numOfTrainningSamples; i++){
		int a = rand() & 1, b = rand() & 1, c = a^b;
		inputVals[0] = a;
		inputVals[1] = b;
		actualVals[0] = c;

		snn.feedForward(inputVals);
		snn.getResults(outputVals);
		snn.backPropogation(actualVals);

		printf("Training: %d ^ %d = %d, out: %f, RA: %f\n", a, b, c, outputVals[0], snn.getRunningAverage());
	}

	system("pause");
} 