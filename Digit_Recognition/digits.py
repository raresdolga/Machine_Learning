import loader
import numpy as np


def cost(images, lables):
    m = len(lables)
    x = np.ones((m, 1))
    #add bias unit to the image matix
    images = np.column_stack((x, images))
    #initialize theta for 2 hidden layers
    theta1 = np.zeros((785, 160))
    theta2 = np.zeros((161, 10))
    #hidden layers will be denoted by a1,a2 ..
    a1 = sigmoid(np.dot(images, theta1))
    a1 = np.column_stack((x, a1))
    a2 = sigmoid(np.dot(a1, theta2))

    y = format_lables(lables)
    J = (1/m)*(np.sum(log_help(a2, y)))
    d1 = np.zeros(np.shape(theta1));
    d2 = np.zeros(np.shape(theta2));

    # back propagation for batch/stochastic gradient

    for i in range(m):
        delta3 = y[i,:] - a2[i,:]
        print (np.shape(delta3))
        delta2 = (np.dot(theta2, delta3))

def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g


def format_lables(lables):
    m = len(lables)
    # 10 is the number of digits we have to recognize 0,2,3,...9
    y = np.zeros((m,10))
    for i in range(len(lables)):
        y[i][lables[i]] = 1
    return y



def main():
    mnData = loader.MNIST('all_Data')
    images, lables = mnData.load_training()
    cost(images,lables)


def log_help(y, param):
    (m,n) = np.shape(y)
    ones = np.ones((m,n))
    h = np.dot(np.transpose(y), np.log(param)) + np.dot((np.transpose(np.subtract(ones , y))), np.log(np.subtract(ones,param)))
    return h

if __name__ == "__main__":main()