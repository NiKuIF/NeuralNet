/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "Net.h"
#include <string>

Net::Net(const std::vector<unsigned> topology){
    
    unsigned numLayers = topology.size();
    
    std::cout << "numLayer: " << numLayers << std::endl;
        
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        
        // the last layer have always 0 outputs
        int numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum +1];
        
        std::cout << "numOutputs: " << numOutputs << std::endl;
           
        // new layer, fill it with neurons
        for(int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a Neuron: " << neuronNum << std::endl;
        }      
    }
    
    m_layers.back().back().setOutputVal(1.0); // for the bias
}

void Net::getResults(std::vector<double>& resultVals) const
{
    resultVals.clear();   
    for(int n = 0; n < m_layers.back().size() - 1; n++)
        resultVals.push_back(m_layers.back()[n].getOutputVal());
}
 
void Net::backProp(const std::vector<double>& targetVals)
{
    Layer& outputLayer = m_layers.back();
    m_error = 0.0;
    
    for(int n = 0; n < outputLayer.size() - 1; n++)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
   
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    // similar to a average calculation
      
    // just for user output
    //std::cout << "smoothing factor: " << m_recentAverangeSmoothingFactor << std::endl;
    m_recentAverangeError = (m_recentAverangeError * m_recentAverangeSmoothingFactor + m_error)
                            / (m_recentAverangeSmoothingFactor + 1.0);
            
    // calculate the gradients that we know the output is to high or to low
    for(int n = 0; n < outputLayer.size() - 1; n++)
        outputLayer[n].calcOutputGradients(targetVals[n]);
    
    // we only want the hidden layer, we have one
    for(int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
    {
        Layer& hiddenLayer = m_layers[layerNum];
        Layer& nextLayer = m_layers[layerNum + 1];
    
        for(int n = 0; n < hiddenLayer.size(); n++)
            hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
    
    // for every layer except input layer, update the weight
    for(int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum -1];
        
        for(int n = 0; n < layer.size() - 1; ++n)
          layer[n].updateInputWeights(prevLayer);  
    } 
}

void Net::feedForward(const std::vector<double>& inputVals)
{
    // check if we have enough inputs for the input layer -> [0]
    assert(inputVals.size() == m_layers[0].size() - 1);
     
    // set outputvalues of the input layer (is not a own layer)
    for(int i = 0; i < inputVals.size(); i++)
        m_layers[0][i].setOutputVal(inputVals[i]);
    
    // forward propagation, start by 1 cause 0 is the input layer
    for(int layerNum = 1; layerNum < m_layers.size(); layerNum++)
    {
        Layer& prevLayer = m_layers[layerNum - 1];
        
        for(int n = 0; n < m_layers[layerNum].size(); n++)
            m_layers[layerNum][n].feedForward(prevLayer);   
            // give the layer to the neuron cause we are fully connected
    }  
}

void Net::printAllConnections()
{
    std::string weight_string;
    
    for(int layer = 0; layer < m_layers.size(); layer++)
    {
        std::cout << "layernum: " << layer << std::endl;
        
        // add layer
        weight_string += std::to_string(layer);
        weight_string += ", ";
        
        for(int n = 0; n < m_layers[layer].size(); n++)
        {
            // add neuron
            weight_string += std::to_string(n);
            weight_string += ", ";
            
            std::cout << "neuron: " << n << std::endl;
            for(int con = 0; con < m_layers[layer][n].m_outputWeights.size(); con++)
            { 
                // add connection(weights)
                weight_string += std::to_string(con);
                weight_string += "(";
                weight_string += std::to_string(m_layers[layer][n].m_outputWeights[con].weight);
                weight_string += ",";
                weight_string += std::to_string(m_layers[layer][n].m_outputWeights[con].deltaWeight);
                weight_string += "),";
               
                std::cout << "con: " << con << ", weight: " << m_layers[layer][n].m_outputWeights[con].weight << ", delta: " << m_layers[layer][n].m_outputWeights[con].deltaWeight << std::endl;
            }        
        }
        weight_string += "\n";
    }
    
    std::cout << "weigh_string: " << weight_string << std::endl;
    
    // you can save the weigth string to a file, and write the weights to 
    // the neurons so that the net dont have to learn again things he already
    // knows -> maybe you should use JSON, but I didn't had enough motivation
}