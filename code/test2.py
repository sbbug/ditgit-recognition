# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
def sigmoid(x):
    '''
    :param x:
    :return:
    '''
    if type(x)!=np.ndarray:
       return 1/(1+math.exp(-x))
    return 1/(1+np.exp(-x))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
xs = np.random.random((N, D_in))
ys = np.random.random((N, D_out))

# Randomly initialize weights
w1 = np.random.random((D_in, H))
w2 = np.random.random((H, D_out))


losses = []
learning_rate = 0.05
for step in range(1000):
    # Forward pass: compute predicted y
    hin = xs.dot(w1)
    hout = sigmoid(hin)
    oin = hout.dot(w2)
    out = sigmoid(oin)

    y_pred = out
    # Compute and print loss
    loss = np.square(y_pred - ys).sum()
    if step % 50 == 0:
        losses.append(loss)
    print(step, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - ys)
    grad_y_pred = sigmoid(grad_y_pred)*(1-sigmoid(grad_y_pred))
    grad_w2 = hout.T.dot(grad_y_pred)

    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = sigmoid(grad_y_pred) * (1 - sigmoid(grad_y_pred))
    grad_w1 = xs.T.dot(grad_h)

    # Update weights
    print(w1.shape)
    print(grad_w1.shape)
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print(losses)
plt.plot(losses)
plt.show()