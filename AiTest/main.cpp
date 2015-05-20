#include <iostream>
#include <vector>
#include <cmath>
#include "SimpleNeuralNetwork.h"

const int numOfTrainningSamples = 1000;
const int numOfTestSamples = numOfTrainningSamples * .15;

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
	printf("Training done, running average: %f\n", snn.getRunningAverage());

	int correct = 0;
	for (int i = 0; i < numOfTestSamples; i++){
		int a = rand() & 1, b = rand() & 1, c = a^b;
		inputVals[0] = a;
		inputVals[1] = b;
		
		snn.feedForward(inputVals);
		snn.getResults(outputVals);
		int d = int(outputVals[0] + 0.5);
		
		printf("Testing: %d ^ %d = %d, out: %d\n", a, b, c, d);
		if (c == d){
			correct++;
		}
	}

	printf("%d out of %d (%3.2f%%) succeeded after traing %d sample data.\n",
		correct, numOfTestSamples, 100.0*correct/numOfTestSamples, numOfTrainningSamples);
	system("pause");
} 