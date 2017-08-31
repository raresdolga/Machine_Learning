import loader
import numpy as np


def cost(images, lables):
    m = len(lables)
    x = np.ones((m, 1))
    # add bias unit to the image matix
    images = np.multiply(images, 1.0 / 255.0)
    images = np.column_stack((x, images))
    # initialize theta for 2 hidden layers
    theta1 = np.random.rand(785, 160)
    theta2 = np.random.rand(161, 10)
    # hidden layers will be denoted by a1,a2 ...
    # a1 is the first hidden layer ( i.e the second layer in the neuronal network
    for iter in range(10):
        a1 = sigmoid(np.dot(images, theta1))
        a1 = np.column_stack((x, a1))
        a2 = sigmoid(np.dot(a1, theta2))
        y = format_lables(lables)
        J = -(1/m)*(np.sum(log_help(y, a2)))
        print(J)
        # this are partial derivatives
        d1 = np.matrix(np.zeros(np.shape(theta1)))
        d2 = np.matrix(np.zeros(np.shape(theta2)))
        # back propagation for batch/stochastic gradient
        ones = np.ones(np.shape(a1))
        for i in range(10):
            # eliminate the bias unit
            delta3 = np.matrix(y[i, :] - a2[i, :])
            delta2 = np.multiply((np.dot(delta3, np.transpose(theta2[1:, :]))), np.multiply(a1[:, 1:], (ones - a1)[:, 1:]))
            d1 += np.dot(np.transpose(images),delta2)
            d2 += np.transpose(np.multiply(np.transpose(delta3), np.matrix(a1[i, :])))
        np.subtract(theta1, 0.1*np.matrix(d1))
        np.subtract(theta2, 0.1*np.matrix(d2))
        #print(0.1*np.matrix(d2))
        #print(theta2)

def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g


def format_lables(lables):
    m = len(lables)
    # 10 is the number of digits we have to recognize 0,2,3,...9
    y = np.zeros((m, 10))
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
    y_s = np.log(np.transpose(np.subtract(ones , y)))
    param_s = np.log(np.subtract(ones,param))
    # solve -inf issue
    y_s[y_s == -np.inf] = -99999999
    param_s[param_s == -np.inf] = -99999999
    h = np.dot(np.transpose(y), np.log(param)) + np.dot(y_s, param_s)
    return h


if __name__ == "__main__":main()