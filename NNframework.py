#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:43:56 2021

@author: miller
"""

#create 3-layer Neural Network
class neuralNetwork:
    #initilize neuralNetwork
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        import numpy as np
        import scipy.special
        
        self.inputNodes   = inputNodes;
        self.hiddenNodes  = hiddenNodes;
        self.outputNodes  = outputNodes;
        self.learningRate = learningRate;
        
        # weighting set - method 1
        #self.Wih = np.random.rand(self.inputNodes, self.hiddenNodes)-0.5;
        #self.Who = np.random.rand(self.hiddenNodes, self.outputNodes)-0.5;
        # weighting set - method 2
        self.Wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), \
                                    (self.hiddenNodes, self.inputNodes));
        self.Who = np.random.normal(0.0, pow(self.outputNodes, -0.5), \
                                    (self.outputNodes, self.hiddenNodes));
        
        # Activate function is  sigmoid function
        self.activateFunction = lambda x: scipy.special.expit(x);
        
        pass;
    
    #train NN
    def train():
        pass
    
    #query NN
    def query(self, inputsList):
        import numpy as np
        # convert inputs to 2D array
        inputs = np.array(inputsList, ndmin=2).T
        print(inputs)
        
        # inputs to hidden layer
        hiddenInputs  = np.dot(self.Wih, inputs);
        hiddenOutputs = self.activateFunction(hiddenInputs);
        
        # hidden layer to output
        finalInputs  = np.dot(self.Who, hiddenOutputs);
        finalOutputs = self.activateFunction(finalInputs);
        
        return finalOutputs;
    
# ==============================================================
# Test neural network
# ==============================================================

inputNodes  = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3

model = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
model.query([1.0, 0.5, -1.5]);



    
    
    
    
    
    
    
    
    
    
    
    
    
    