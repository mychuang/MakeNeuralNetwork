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
        
    #plotting check
    def plotSingleImage(self, image, titleWord):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        plt.imshow(image, cmap=plt.cm.gray, interpolation="None")
        plt.title(titleWord)

        pass
        

#%% === import MNIST DATA ===
from keras.datasets import mnist
import numpy as np

(KerasTrainImages, KerasTrainLabels), (KerasTestImages, KerasTestLabels) = \
                                      mnist.load_data()

# === Process Trainning Data ===
trainSignals = (KerasTrainImages/255 * 0.99) + 0.01
trainTargets = np.zeros((60000, 10))
for index in range(len(trainTargets)):
    trainTargets[index, int(KerasTrainLabels[index])] = 0.99
# === Process Testing Data ===    
testSignals = (KerasTestImages/255 * 0.99) + 0.01
testTargets = np.zeros((10000, 10))
for index in range(len(testTargets)):
    testTargets[index, int(KerasTestLabels[index])] = 0.99

#%% The Loop for trainning
print("Build Neural Network")
inputNodes   = 784
hiddenNodes  = 100
outputNodes  = 10
learningRate = 0.1
epochs       = 10
performance  = np.array([])
trainScore   = np.array([])
model = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

print(" == Training == ")
for times in range(epochs):
    for index in range(60000):
        signal = trainSignals[index].reshape(inputNodes)
        model.train(signal, trainTargets[index])
        result = model.query(signal)
        if(np.argmax(result) == np.argmax(trainTargets[index])):
            trainScore = np.append(trainScore, 1)
        else:
            trainScore = np.append(trainScore, 0)
    performance = np.append(performance, trainScore.sum()/float(len(trainScore)))
    print("Trainning Performance ", times, ": ", trainScore.sum()/float(len(trainScore)))
    
#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
plt.plot(performance)
plt.title("Performance in 10 epochs")
plt.show()

#%% The loop for testing
print(" == Testing ==")
testScore = np.array([])

for index in range(10000):
    signal = testSignals[index].reshape(inputNodes)
    result = model.query(signal)
    if(np.argmax(result) == np.argmax(testTargets[index])):
        testScore = np.append(testScore, 1)
    else:
        testScore = np.append(testScore, 0)
print("Testing Performance: ", testScore.sum()/float(len(testScore)))   


        




    
    
    
    
    
    
    
    
    
    
    
    
    
    