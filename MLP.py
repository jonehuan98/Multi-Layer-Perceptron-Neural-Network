import numpy as np
#from random import random
import matplotlib.pyplot as plt

class MLP(object):

    def __init__(self, numInputs=2, hiddenLayers=[2], numOutputs=1):
        self.numInputs = numInputs
        self.hiddenLayers = hiddenLayers
        self.numOutputs = numOutputs
        
        layers = [numInputs] + hiddenLayers + [numOutputs]
        
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # bias = []
        # for i in range(len(hiddenLayers)):
        #     b = np.random.rand(1, hiddenLayers[i])
        #     bias.append(b)
        # bias.append(np.random.rand(1, numOutputs))
        # self.bias = bias     
        
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoidDerivative(self, x):
        return x * (1.0 - x)

    def relu(self, x):
        return np.maximum(x, 0)

    def reluDerivative(self, x):
        return x > 0 #returns 1 if x>0, otherwise return 0

    def forwardProp(self, inputs):
        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):            
            net_inputs = (np.dot(activations, w))
            # activation function can be switched out
            activations = self.sigmoid(net_inputs)
            self.activations[i + 1] = activations
        
        return activations


    def backProp(self, error):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            # activation function derivative can be switched out
            delta = error * self.sigmoidDerivative(activations)
            deltaReshape = delta.reshape(delta.shape[0], -1).T
            currentActivations = self.activations[i]
            currentActivations = currentActivations.reshape(currentActivations.shape[0],-1)
            self.derivatives[i] = np.dot(currentActivations, deltaReshape)
            
            # backpropogate
            error = np.dot(delta, self.weights[i].T)

    def gradientDescent(self, learningRate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
   
    def meanSquareError(self, target, output):
        return np.average((target - output) ** 2)

    def train(self, inputs, targets, epochs, learningRate):
        for i in range(epochs):
            errorSum = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forwardProp(input)
                error = target - output
                self.backProp(error)
                # gradient descent and update weights
                self.gradientDescent(learningRate)
                errorSum += self.meanSquareError(target, output)
                
            epochError = [errorSum/ len(samples)]
            errorList.append(epochError)
            
        print("Final error:", epochError)
    
    def accuracy(self, predicted, target):
        accuracy = np.sum(predicted == target)/ len(target)
        return accuracy

    def predict(self, inputs, targets):
        output = self.forwardProp(inputs)
        predictions = np.where(output>=0.5, 1, 0)
        #print("predictions:", predictions)
        accuracy = self.accuracy(predictions, targets)
        print("Accuracy", accuracy)

if __name__ == "__main__":
    np.random.seed(100)

    #XOR problem
    # 1 1 > 0
    # 0 0 > 0
    # 1 0 > 1
    # 0 1 > 1
    samples = np.array([[0,0], [0,1], [1,0], [1,1]])
    targets = np.array([[0,1,1,0]]).T
    iterations = 20000
    learningRate = 0.1
    errorList = []
    mlp = MLP()

    print("Initial Weights:", mlp.weights)
    mlp.train(samples, targets, iterations, learningRate)
    
    print("Final Weights:", mlp.weights)
    mlp.predict(samples, targets)


    #plotting
    x = np.array(range(iterations))
    plt.figure()
    plt.plot(x,errorList)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("[XOR], Learning rate:" + str(learningRate) + ", Activation: Sigmoid" )
    plt.show
