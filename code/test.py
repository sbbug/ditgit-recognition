import numpy as np
import math
import matplotlib.pyplot as plt


#定义激活函数,可以计算的那个元素以及数组
def sigmoid(x):
    '''
    :param x:
    :return:
    '''
    if type(x)!=np.ndarray:
       return 1/(1+math.exp(-x))
    return 1/(1+np.exp(-x))

#激活函数的偏导数
def sigDer(x):
    '''
    :param x:
    :return:
    '''
    return sigmoid(x)*(1-sigmoid(x))

#实现向量与矩阵相乘
def dot(w,x):
    '''
    :param w:
    :param x:
    :return:
    '''
    #先将行向量转换为列向量
    x = np.transpose(x)
    #计算相乘
    return np.transpose(np.dot(w,x))

#将计算获取的向量进行激活
def sigmoidW(x):
    '''
    :param w:
    :return:
    '''
    return sigmoid(x)

#计算输出层的实际值与深数据真实值的差
def minus(Y_,Y):
    '''
    :param Y_: 实际值
    :param Y: 期望值
    :return:
    '''
    return (Y_ - Y)
#计算实际值与期望值之间的误差
def loss(Y_,Y):
    '''
        :param Y_: 实际值
        :param Y: 期望值
        :return:
    '''
    return np.power(minus(Y_,Y),2)
#计算w2的偏导数，并返回
def wODer(Y_,Y,out,hout):

    w = np.ones((2,2))
    return w*minus(Y_,Y)*sigDer(out)*sigmoid(hout)

#实现矩阵更新
def updateW(w,w_,k):
    '''
    :param w: w是原矩阵
    :param w_: w_是偏导数矩阵
    :param k: k是更新幅度，也是学习速率
    :return:
    '''
    return (w-k*w_)

def wHDer(Y_,Y,out,w2,hout,X):

    w = np.ones(w2.shape)

    for i in range(len(w)):
        for j in range(len(w[i])):
            t = float(0.0)
            for n in range(len(Y_)):
                t = t+(Y_[n]-Y[n])*sigDer(out[n])*w2[n][i]*X[j]

            w[i][j]=t
    return w

if __name__ == "__main__":

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
    learning_rate=0.005

    for step in range(1000):
            #计算h层输出
            hin = xs.dot(w1)
            # print("h层计算输出")
            # print(hin)

            #将输出的h层激活
            hout = sigmoidW(hin)
            # print("h激活输出")
            # print(hout)

            #将h层数据传到o层，并计算输出
            oin = hout.dot(w2)
            # print("o层输出")
            # print(oin)

            #将输出的o层激活
            out = sigmoidW(oin)
            Y_ = out
            # print("o层激活输出")
            # print(out)

            print("损失")
            print(round(np.square(Y_ - ys).sum(),6))
            if step%10==0:
               losses.append(round(np.square(Y_ - ys).sum(),6))

            grad_y_pred =  2*(Y_ - ys)
            grad_o_sig = sigDer(out)
            grad_o_relu = grad_y_pred*grad_o_sig
            grad_w2 = hout.T.dot(grad_o_relu)

            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h_sig = sigDer(hout)
            grad_h_relu=grad_h_relu*grad_h_sig
            grad_w1 = xs.T.dot(grad_h_relu)
            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

    print(losses)
    plt.plot(losses)
    plt.show()