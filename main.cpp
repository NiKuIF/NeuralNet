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
#include "Neuron.h"
#include "Net.h"


void showVectorVals(std::string label, std::vector<double> &v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }  
    std::cout << std::endl;
}

int main(int argc, char** argv) {
 
    // learn the Net the work of a AND-Function   
    // 00 -> 0
    std::vector<double> input00;
    input00.push_back(0.0);
    input00.push_back(0.0);
    std::vector<double> output00;
    output00.push_back(0.0);
    
    // 11 -> 1
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
    
    // for a AND-Gate a 2 1 Topology would be enough
    
    // simple 2 3 1 Topology
    std::vector<unsigned> topology;
    topology.push_back(2); 
    topology.push_back(3); 
    topology.push_back(1); 
    
    Net myNet(topology);
    
    std::vector<double> inputVals, targetVals, resultVals;
    
    for(int i = 0; i < 2000; i++ ){
    
        std::cout << std::endl << "Pass " << i <<  std::endl;
       
       if(i%4 == 0){
           inputVals = input11;
           targetVals = output11;
       }
       else if(i%3 == 0) {          
           inputVals = input10;
           targetVals = output10;
       }
       else if(i%2 == 0){
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
 
    // print the connection of all neurons
    std::cout << "\n start printing weights: \n" << std::endl;
    myNet.printAllConnections();
 
    return 0;
}

