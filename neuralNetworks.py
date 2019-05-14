
import numpy as np
import os
from tensorflow import keras


# Neural Network class definition
class NeuralNetwork():
    def __init__(self, Input, Output, ):
        # Save all variables in self for future references
        self.Input = Input
        self.Output = Output
        self.input_size=784
        self.hidden_size = 128
        self.output_size = 10
        self.learning_rate=0.001

        # initialize the weights
        self.W_0 = 2*np.random.random((self.hidden_size, self.input_size))-1
        self.b_0 = 2*np.random.random((self.hidden_size))-1
        self.W_1 = 2*np.random.random((self.output_size, self.hidden_size))-1
        self.b_1 = 2*np.random.random((self.output_size))-1

    # Sigmoid function gives a value between 0 and 1

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def forward(self, x):
        # Multiply the input with weights and find its sigmoid activation for all layers
        hidden_layer = self.sigmoid(np.dot(self.W_0, x)+self.b_0)
        output_layer = self.softmax(np.dot(self.W_1, hidden_layer)+self.b_1)
        return output_layer

    def train(self, epoches):
        for e in range(epoches):
            for i in range(self.Input.shape[0]):

                x=(self.Input[i]).flatten()
                y=np.zeros(10)
                y[Output[i]]=1

                hidden_layer = self.sigmoid(np.dot(self.W_0, x) + self.b_0)
                output_layer = self.softmax(np.dot(self.W_1, hidden_layer) + self.b_1)

                # Calculate delta for parameters
                delta_b0 = np.multiply(np.dot(self.W_1.T,output_layer-y),np.multiply(hidden_layer,1-hidden_layer))
                delta_W0 = np.dot(delta_b0.reshape(self.hidden_size,1),x.reshape(1,self.input_size))
                delta_b1 = output_layer-y
                delta_W1 = np.dot(delta_b1.reshape(self.output_size,1), hidden_layer.reshape(1,self.hidden_size))

                # back propagation
                self.W_0 -= self.learning_rate*delta_W0
                self.b_0 -= self.learning_rate*delta_b0
                self.W_1 -= self.learning_rate*delta_W1
                self.b_1 -= self.learning_rate*delta_b1


if __name__ == '__main__':

    if not os.path.exists("train_images.npy"):
        print("here")
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        np.save("train_images.npy", train_images)
        np.save("train_labels.npy",train_labels)
        np.save("test_images.npy", test_images)
        np.save("test_labels.npy",test_labels)
    else:

        train_images = np.load("train_images.npy")
        train_labels = np.load("train_labels.npy")
        test_images = np.load("test_images.npy")
        test_labels = np.load("test_labels.npy")

    Input = train_images
    Output = train_labels
    neural_network = NeuralNetwork(Input, Output)
    neural_network.train(5)
    accuracy=0
    for i in range(test_images.shape[0]):
        x=test_images[i].flatten()
        y=np.argmax(neural_network.forward(x))
        if y==test_labels[i]:
            accuracy+=1
            #print(neural_network.forward(x))
    print(accuracy)

    accuracy=accuracy/test_images.shape[0]
    print(accuracy)


