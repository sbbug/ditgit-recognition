'''
对之前写的bp神经网络进一步优化，进行封装
'''
import numpy as np
import math
#定义层
class BP:

    #构造方法
    def __init__(self,
                 sample_n=0,
                 input_n=1,
                 hidden_n=1,
                 sample_out=1,
                 train_iteration=1000,
                 x_train=[],
                 y_train=[],
                 learning_rate=0.05,
                 loss=True):
        '''
         :param sample_n: 批量样本数量
        :param input_n: 单个样本输入维度
        :param hidden_n: 隐藏层节点数
        :param sample_out: 输出维度
        :param train_iteration: 迭代次数
        :param x_train: 训练集输入数据
        :param y_train: 训练集输出数据
        :param learning_rate: 学习速率
        :param loss: 损失
        '''
        self.sample_n = sample_n
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.sample_out = sample_out
        self.train_iteration = train_iteration
        self.learning_rate = learning_rate
        self.loss = loss
        #训练数据加载
        print(len(x_train))
        if self.sample_n!=len(x_train):
            raise RuntimeError('sample_n !=len(x_train)')

        self.x_train = x_train
        self.y_train = y_train

        self.init()

    #参数初始化
    def init(self):
        '''
        :return:
        '''
        self.input_hidden_w = np.random.random((self.input_n,self.hidden_n))
        self.hidden_output_w = np.random.random((self.hidden_n, self.sample_out))

        self.input_in = None
        self.hidden_in = None
        self.hidden_out = None
        self.output_in = None
        self.output_out = None

        self.output_y = None

        #定义损失集合
        self.losses = []

    #前向传播
    def forward(self):

        self.hidden_in = self.x_train.dot(self.input_hidden_w)
        self.hidden_out = self.sigmoidW(self.hidden_in)

        self.output_in = self.hidden_out.dot(self.hidden_output_w)
        self.output_out = self.sigmoidW(self.output_in)

        self.output_y = self.output_out

    #实现单样本预测
    def predictSmaple(self,x):

        self.hidden_in = x.dot(self.input_hidden_w)
        self.hidden_out = self.sigmoidW(self.hidden_in)

        self.output_in = self.hidden_out.dot(self.hidden_output_w)
        self.output_out = self.sigmoidW(self.output_in)

        self.output_y = self.output_out

        return self.output_y
    #反向更新
    def backUpdate(self):

        if self.loss:
            print(round((np.square(self.output_y - self.y_train).sum()) / self.input_n, 6))
            self.losses.append(round((np.square(self.output_y - self.y_train).sum()) / self.input_n, 6))


        # 计算w2的梯度损失
        grad_y_pred = (2 * (self.output_y - self.y_train))
        grad_o_sig = self.sigDer(self.output_out)
        grad_o = grad_y_pred * grad_o_sig
        grad_w2 = self.hidden_out.T.dot(grad_o)

        # 计算w1的梯度损失
        grad_h_relu = grad_y_pred.dot(self.hidden_output_w.T)
        grad_h_sig = self.sigDer(self.hidden_out)
        grad_h = grad_h_relu * grad_h_sig
        grad_w1 = self.x_train.T.dot(grad_h)
        # 更新梯度
        self.input_hidden_w -= self.learning_rate*grad_w1
        self.hidden_output_w -= self.learning_rate*grad_w2

    #开始训练
    def start(self):
        for i in range(self.train_iteration):
            self.forward()
            self.backUpdate()

    #获取损失数据集合
    def getLosses(self):

        return self.losses

    # 激活函数
    def sigmoid(self,x):
        '''
        :param x:
        :return:
        '''
        if type(x) != np.ndarray:
            return 1 / (1 + math.exp(-x))
        return 1 / (1 + np.exp(-x))

    # 激活函数的偏导数
    def sigDer(self,x):
        '''
        :param x:
        :return:
        '''
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # 将计算获取的向量进行激活
    def sigmoidW(self,x):
        '''
        :param w:
        :return:
        '''
        return self.sigmoid(x)

    #保存训练好的模型
    def saveModel(self,model_dir):

        file_w1 = model_dir+"/w1.txt"
        file_w2 = model_dir+"/w2.txt"

        np.savetxt(file_w1, self.input_hidden_w)
        np.savetxt(file_w2,self.hidden_output_w)

        print("模型保存成功")

    def loadModel(self,model_dir):
        file_w1 = model_dir + "/w1.txt"
        file_w2 = model_dir + "/w2.txt"

        self.input_hidden_w=np.loadtxt(file_w1)
        self.hidden_output_w=np.loadtxt(file_w2)

        print("模型载入成功")
    #加载训练好的模型




