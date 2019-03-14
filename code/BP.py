# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import LoadData as rd
import cv2 as cv

#激活函数
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
#将计算获取的向量进行激活
def sigmoidW(x):
    '''
    :param w:
    :return:
    '''
    return sigmoid(x)

#向量小数点截取
def roundVector(vector):
    for i in range(len(vector)):
        vector[i] = round(vector[i],2)

    return vector

#神经网络前向传播过程
def forward(x,w1,w2):
    hin = x.dot(w1)
    # 将输出的h层激活
    hout = sigmoidW(hin)
    # 将h层数据传到o层，并计算输出
    oin = hout.dot(w2)
    # 将输出的o层激活
    out = sigmoidW(oin)

    return out

#保存模型数据
def saveModel(file,w):

    np.savetxt(file,w)
#加载模型
def loadModel(file):

    return np.loadtxt(file)

def getImageData(file):

    img = cv.imread(file)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_resize = cv.resize(img_gray, (32, 32))
    ret, img_binary = cv.threshold(img_resize, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    w,h = img_binary.shape
    vector = np.ones(w*h)

    for i in range(len(img_binary)):
        for j in range(len(img_binary[i])):
            if img_binary[i][j] == 255:
                img_binary[i][j] = 0
            elif img_binary[i][j] == 0:
                img_binary[i][j] = 1

            vector[i*32+j] = img_binary[i][j]

    return vector

#定义网络初始参数
'''
N:批量样本数
D_in:输入向量维数，样本特征数
H:隐藏层数目
D_out:输出向量维数
'''
if __name__ =="__main__":
    N, D_in, H, D_out = 1934, 1024, 64, 10

    x_train,y_train = rd.getData("../data/trainingDigits")

    #参数初始化
    w1 = np.random.random((D_in, H))
    w2 = np.random.random((H, D_out))

    #存放损失数据
    losses = []
    #学习速率
    learning_rate = 0.05

    #批量样本更新策略
    miniBatch = 100

    for step in range(1000):
        # 计算h层输出
        hin = x_train.dot(w1)
        # print("h层计算输出")
        # print(hin)

        # 将输出的h层激活
        hout = sigmoidW(hin)
        # print("h激活输出")
        # print(hout)

        # 将h层数据传到o层，并计算输出
        oin = hout.dot(w2)
        # print("o层输出")
        # print(oin)

        # 将输出的o层激活
        out = sigmoidW(oin)
        Y_ = out
        # print("o层激活输出")
        # print(out)

        print(round((np.square(Y_ - y_train).sum())/N, 6))
        if step % 100 == 0:
            losses.append(round((np.square(Y_ - y_train).sum())/N, 6))

        #计算w2的梯度损失
        grad_y_pred = (2 * (Y_ - y_train))
        grad_o_sig = sigDer(out)
        grad_o = grad_y_pred * grad_o_sig
        grad_w2 = hout.T.dot(grad_o)

        #计算w1的梯度损失
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h_sig = sigDer(hout)
        grad_h = grad_h_relu * grad_h_sig
        grad_w1 = x_train.T.dot(grad_h)
        # 更新梯度
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


    #使用训练集进行测试
    x_test,y_test = rd.getData("../data/testDigits")

    right_n = 0
    all_n = len(x_test)
    for i in range(len(x_test)):

        Y_ = forward(x_test[i],w1,w2)
        if np.argmax(Y_)==np.argmax(y_test[i]):
            right_n=right_n+1




    print("总数---"+str(all_n))
    print("正确个数"+str(right_n))
    print("准确率----")
    print(float(right_n/all_n))

    #训练好的模型保存
    saveModel("./model/w1.txt",w1)
    saveModel("./model/w2.txt",w2)

    print(losses)
    plt.plot(losses)
    plt.show()

'''
网络描述:
1024个输入节点，64个隐藏节点，10个输出节点
隐含层与输出层均使用sigmoid激活

效果:
对于已经有的数据集训练效果很好，对自己制作的新的图片效果很差
通用性差
'''