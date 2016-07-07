/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Net.h
 * Author: kurtnistelberger
 *
 * Created on July 7, 2016, 5:55 PM
 */

#ifndef NET_H
#define NET_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include "Neuron.h"

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


#endif /* NET_H */

