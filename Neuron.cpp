#include "Neuron.h"

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) : 
    eta(0.15), alpha(0.5) {
    
    for(unsigned c = 0; c < numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight(); // start with a random weight
    }
    m_myIndex = myIndex;  
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
    for(int n = 0; n < prevLayer.size(); n++) {
        
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight = 
        eta     // learning rate
        * neuron.getOutputVal() 
        * m_gradient
        + alpha
        * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;
    
    for(int n = 0; n < nextLayer.size() - 1; n++)
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    
    return sum;     
}

void Neuron::calcHiddenGradients(const Layer& nextLayer) {

    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals)
{
    double delta = targetVals - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
    
    /*
        negative: output to big
        positive: output to small
     */
}

double Neuron::transferFunction(double x) 
{   // output range -1 to 1    
    return tanh(x); 
}

double Neuron::transferFunctionDerivative(double x) 
{
   return 1.0 - x*x;
}

void Neuron::feedForward(Layer &prevLayer) {
    
    double sum = 0.0;
    for(unsigned n = 0; n < prevLayer.size(); ++n)
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    
    m_outputVal = transferFunction(sum);  
}
