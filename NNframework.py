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
    def train(self, inputsList, targetList):
        import numpy as np
        # === calculate signals ====
        # convert inputs to 2D array
        inputs  = np.array(inputsList, ndmin=2).T
        targets = np.array(targetList, ndmin=2).T
        
        # clac signal into hidden layer
        hiddenInputs  = np.dot(self.Wih, inputs);
        hiddenOutputs = self.activateFunction(hiddenInputs);
        
        # calc signal into output layer
        finalInputs  = np.dot(self.Who, hiddenOutputs);
        finalOutputs = self.activateFunction(finalInputs);
        
        # === claculate error & back propagation ===
        outputError = targets - finalOutputs;
        hiddenError   = np.dot(self.Who.T, outputError);
        
        # === Update weighting ===
        self.Who += self.learningRate * \
            np.dot((outputError * finalOutputs * (1.0-finalOutputs)), \
                   np.transpose(hiddenOutputs))
        self.Wih += self.learningRate * \
            np.dot((hiddenError * hiddenOutputs * (1.0-hiddenOutputs)), \
                   np.transpose(inputs))
        
        pass
    
    #query NN
    def query(self, inputsList):
        import numpy as np
        # convert inputs to 2D array
        inputs = np.array(inputsList, ndmin=2).T
        
        # inputs to hidden layer
        hiddenInputs  = np.dot(self.Wih, inputs);
        hiddenOutputs = self.activateFunction(hiddenInputs);
        
        # hidden layer to output
        finalInputs  = np.dot(self.Who, hiddenOutputs);
        finalOutputs = self.activateFunction(finalInputs);
        
        return finalOutputs;

#%% === import MNIST DATA ===
from keras.datasets import mnist

(KerasTrainImages, KerasTrainLabels), (KerasTestImages, KerasTestLabels) = \
                                      mnist.load_data()

#%% === Randomly get test data from MNIST ===
import numpy as np
import random
test50 = np.zeros((50, 28, 28))
tlab50 = np.zeros(50)

for times in range(50):
    index = random.randrange(10000)
    test50[times] = KerasTestImages[index]
    tlab50[times] = KerasTestLabels[index]
#%% === Process Data ===
signals = (test50/255 * 0.99) + 0.01
targets = np.zeros((50, 10))
for index in range(len(targets)):
    targets[index, int(tlab50[index])] = 0.99

#%% plotting check
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
plt.imshow(signals[0], cmap=plt.cm.gray, interpolation="None")
plt.title("Plot 2D array")

#%% === NN setting 
inputNodes   = 784
hiddenNodes  = 100
outputNodes  = 10
learningRate = 0.3 

 





    
    
    
    
    
    
    
    
    
    
    
    
    
    