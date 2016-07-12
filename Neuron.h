/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Neuron.h
 * Author: kurtnistelberger
 *
 * Created on July 7, 2016, 5:46 PM
 */

#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};

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
    
    // make public, so that I can print it easily
    std::vector<Connection> m_outputWeights;
    
private:  
    double eta;
    /*
     0.0 - slow learner
     0.2 - medium learner
     1.0 - reckless learner
     */
    
    double alpha; // multiplier of last weight change (momentum)
    
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand()/double(RAND_MAX);}
    
    double sumDOW(const Layer &nextLayer) const;
    
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
};


#endif /* NEURON_H */

