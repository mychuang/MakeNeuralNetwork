# MakeNeuralNetwork
## The following step is about how to construct 3-layer neural network by python.
### step 1
At beginning, we construct a class, which represent our NN.

```python
class neuralNetwork:
  def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
  pass

  def train(self, inputsList, targetList):
  pass

  def query(self, inputsList):
  pass
```
- inputNodes: the number of input nodes 
- hiddenNodes: the number of hidden nodes
- outputNodes: the number of output nodes
- learningRate: learning rate
<hr>

### step 2
And now, create the intial setting of NN
```python
class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        import numpy as np
        import scipy.special
        self.inputNodes   = inputNodes;
        self.hiddenNodes  = hiddenNodes;
        self.outputNodes  = outputNodes;
        self.learningRate = learningRate;

        self.Wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), \
                self.hiddenNodes, self.inputNodes));
        self.Who = np.random.normal(0.0, pow(self.outputNodes, -0.5), \
                (self.outputNodes, self.hiddenNodes));

        self.activateFunction = lambda x: scipy.special.expit(x);

```
- The first guess of weighting use **np.random.normal()** function, 0.0 represent the mean of normal distribution is 0.
- The weighting formula **1/(the num of input nodes)^0.5**
- The activate function use sigmoid function, which can use the **cipy.special.expit()** method.
<hr>

### step 3
Define the query function, which can be obtain the result after trainning
```python
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
```
Just follow the basic fomula: Y = W * X, then O = sigmoid(X)
<hr>

### step 4
The trainning function can see the annotation, it is also similar to the query function.
The only special point is about how to update the weighting.<br>
See the code below:
```python
self.Who += self.learningRate * \
    np.dot((outputError * finalOutputs * (1.0-finalOutputs)), np.transpose(hiddenOutputs))

self.Wih += self.learningRate * \
    np.dot((hiddenError * hiddenOutputs * (1.0-hiddenOutputs)), np.transpose(inputs))
```
It just the result to combine back-propagation and gradient descend method.
The fomula like:<br>
**delt(W)=lr * E * sigmoid(O) * (1-sigmoid(O)) * O**<br>
O means output.

