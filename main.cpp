/* 
 * File:   main.cpp
 * Author: NiKu
 * https://vimeo.com/19569529
 * Created on June 26, 2015, 11:18 PM
 */

#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val){m_outputVal = val;}
    double getOutputVal (void) const {return m_outputVal;}
    void feedForward(Layer &prevLayer);
    void calcOutputGradients(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    
private:  
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand()/double(RAND_MAX);}
    
    double sumDOW(const Layer &nextLayer) const;
    
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer& prevLayer)
{
    for(unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight = 
        eta
        * neuron.getOutputVal()
        * m_gradient
        + alpha
        * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    
    for(unsigned n = 0; n<nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;     
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    
double dow = sumDOW(nextLayer);
m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
    
}

void Neuron::calcOutputGradients(double targetVals)
{
    double delta = targetVals - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) 
{
    // output range -1 - 1    
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) 
{
   return 1.0 - x*x;
}

void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;
    
    for(unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() * 
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = Neuron::transferFunction(sum);  
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;  
}

class Net
{
public:
    Net(const std::vector<unsigned> topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double>  &targetVals);
    void getResults(std::vector<double>  &resultVals) const;
    double getRecentAverageError() const {return m_recentAverangeError; }
    
private:
    std::vector<Layer> m_layers;    
    double m_error;
    double m_recentAverangeError;
    double m_recentAverangeSmoothingFactor;
};

void Net::getResults(std::vector<double>  &resultVals) const
{
    resultVals.clear();   
    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }   
}
 
void Net::backProp(const std::vector<double> &targetVals)
{
    Layer & outputLayer = m_layers.back();
    m_error = 0.0;
    
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
   
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
      
    m_recentAverangeError = (m_recentAverangeError * m_recentAverangeSmoothingFactor + m_error)
                            / (m_recentAverangeSmoothingFactor + 1.0);
            
    for(unsigned n = 0; n< outputLayer.size() -1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
    
        for(unsigned n = 0; n<hiddenLayer.size(); n++)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    
    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum -1];
        
        for(unsigned n = 0; n<layer.size() -1;++n)
        {
          layer[n].updateInputWeights(prevLayer);   
        }        
    } 
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);
    for(unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    // forward prop
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++ layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned n = 0; n < m_layers[layerNum].size(); ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }  
}

Net::Net(const std::vector<unsigned> topology){
    
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum +1];
        // new layer, fill it with neurons
    
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a Neuron" << std::endl;
        }      
    }
    
    m_layers.back().back().setOutputVal(1.0); // for the bias
}

void showVectorVals(std::string label, std::vector<double> &v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }  
    std::cout << std::endl;
}

int main(int argc, char** argv) {
 
    // inputs for XOR-Function
    /*   
    // learn the Net the work of a XOR-Function   
    // 00 -> 0
    std::vector<double> input00;
    input00.push_back(0.0);
    input00.push_back(0.0);
    std::vector<double> output00;
    output00.push_back(0.0);
    
    // 11 -> 0
    std::vector<double> input11;
    input11.push_back(1.0);
    input11.push_back(1.0);
    std::vector<double> output11;
    output11.push_back(0.0);
    
    // 10 -> 1
    std::vector<double> input10;
    input10.push_back(1.0);
    input10.push_back(0.0);
    std::vector<double> output10;
    output10.push_back(1.0);
    
    // 01 -> 1
    std::vector<double> input01;
    input01.push_back(0.0);
    input01.push_back(1.0);
    std::vector<double> output01;
    output01.push_back(1.0);
  */  
    
    // learn the Net the work of a AND-Function   
    // 00 -> 0
    std::vector<double> input00;
    input00.push_back(0.0);
    input00.push_back(0.0);
    std::vector<double> output00;
    output00.push_back(0.0);
    
    // 11 -> 0
    std::vector<double> input11;
    input11.push_back(1.0);
    input11.push_back(1.0);
    std::vector<double> output11;
    output11.push_back(1.0);
    
    // 10 -> 0
    std::vector<double> input10;
    input10.push_back(1.0);
    input10.push_back(0.0);
    std::vector<double> output10;
    output10.push_back(0.0);
    
    // 01 -> 0
    std::vector<double> input01;
    input01.push_back(0.0);
    input01.push_back(1.0);
    std::vector<double> output01;
    output01.push_back(0.0);
    
    
    
    std::vector<unsigned> topology;

    // simple 2 4 1 Topology
    topology.push_back(2); 
    topology.push_back(5); 
    topology.push_back(1); 
    
    Net myNet(topology);
    
    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    int while_counter = 1;
    while (true){
       ++ trainingPass;
       std::cout << std::endl << "Pass " << trainingPass;
       
       ++while_counter;
       
       if(while_counter > 2000){
           break;
       }
       
       if(while_counter%4 == 0){
           inputVals = input11;
           targetVals = output11;
       }
       else if(while_counter%3 == 0) {          
           inputVals = input10;
           targetVals = output10;
       }
       else if(while_counter%2 == 0){
           inputVals = input01;
           targetVals = output01;
       }
       else {
           inputVals = input00;
           targetVals = output00;
       }

       showVectorVals(": Inputs:", inputVals);
       myNet.feedForward(inputVals);
       
       myNet.getResults(resultVals);
       showVectorVals("Outputs:", resultVals);
       
       showVectorVals("Targets:", targetVals);
       assert(targetVals.size() == topology.back());
       
       myNet.backProp(targetVals);
       
       std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
          
    }

    std::cout << std::endl << "Done" << std::endl;
    
    return 0;
}

