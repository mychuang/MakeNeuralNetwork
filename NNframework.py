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
            
        pass;
    
    #train NN
    def train():
        pass
    
    #query NN
    def query():
        pass