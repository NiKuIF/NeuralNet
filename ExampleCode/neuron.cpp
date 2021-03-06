#include "neuron.h"

//double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
//double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(unsigned numberOutputs, unsigned index){
	for (unsigned c = 0; c < numberOutputs; c++){
		outputWeights.push_back(Connection());
	}
	this->index=index;
}

Neuron::Neuron(unsigned numberOutputs, unsigned index,  const string &transferFunction){
	for (unsigned c = 0; c < numberOutputs; c++){ 
		outputWeights.push_back(Connection());
	}
	this->index=index;
	this->transferFunction=transferFunction;
}

void  Neuron::setOutputValue(double value) {
  outputValue = value; 
}

double  Neuron::getOutputValue() const{
  return outputValue; 
}

double Neuron::transferFunctionTanH(double x){
	// tanh - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::transferFunctionTanHDerivative(double x){
	// tanh derivative aproximation
	//return 1.0 - x * x;
	
	return 1.0 - tanh(x) * tanh(x);
}

double Neuron::transferFunctionSig(double x){
	// tanh - output range [-1.0..1.0]
    return 1/(1+exp(-1*x));
}

double Neuron::transferFunctionSigDerivative(double x){
	// tanh derivative aproximation
	return exp(-1*x)/pow((1+exp(-1*x)),2);
}

double Neuron::transferFunctionStep(double x){
	return (x<0)?0:x;
}

double Neuron::transferFunctionStepDerivative(){
	// step derivative aproximation
	return 1.0 ;
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum=0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
	for (unsigned n = 0; n < prevLayer.size();  n++){
		sum+=prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[index].weight;
	}

	// activate function or transfer /sig /gaussian /linear/ step
	if (this->transferFunction=="th")
		outputValue = Neuron::transferFunctionTanH(sum);
	else if (this->transferFunction=="sig")
		outputValue = Neuron::transferFunctionSig(sum);

}

void Neuron::calculateOutputGradients(double targetValue){
	double delta= targetValue - outputValue;
	if (this->transferFunction=="th")
		gradient =delta * Neuron::transferFunctionTanHDerivative(outputValue);
 	else if (this->transferFunction=="sig")
		gradient =delta * Neuron::transferFunctionSigDerivative(outputValue);
}

double Neuron::sumDOW(const Layer &nextLayer) const{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed.
	for(unsigned n = 0; n < nextLayer.size() -1; n++){
		sum+=outputWeights[n].weight * nextLayer[n].gradient;
  	}

  	return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer){
	double dow= sumDOW(nextLayer);
	gradient = dow * Neuron::transferFunctionTanHDerivative(outputValue);
}

void Neuron::updateInputWeights(Layer &prevLayer, const double &eta, const double &alpha){
 	// 	The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
	for(unsigned n = 0; n < prevLayer.size(); n++){
		Neuron &neuron = prevLayer[n];
		Connection &conn=neuron.outputWeights[index];
		
		double oldDeltaWeight = conn.deltaWeight;
		double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputValue()
                * gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        conn.deltaWeight = newDeltaWeight;
        conn.weight += newDeltaWeight;
  	}

}