import os
import numpy as np
'''
图像数据预处理模块，将图像数据与标签转换为numpy数组
'''

def getData(file):

    #定义存放图像数据的矩阵
    imgDatas = []
    #定义存放数据的标签
    imgLabels = []
    #获取目录下所有文件夹名字
    file_names = os.listdir(file)

    for f in file_names:

        #将图像对应的标签数据转换为向量
        label = np.zeros((10))
        label[int(f[0])]=float(1)
        imgLabels.append(label)

        #获取图像数据
        file_img = open(file+"/"+f)
        line = file_img.readline()

        img_data = ''
        while line:
            line = line.replace('\n','') #去掉字符串换行符
            img_data=img_data+line #将每个文件里读取的数据拼接成一个字符串，方便转为向量
            line = file_img.readline()

        #将字符串列表转换为浮点数列表
        img_data = map(float,img_data)
        img_data = list(img_data)

        imgDatas.append(img_data)

    return np.array(imgDatas),np.array(imgLabels)

def getImg(file):

    # 将图像对应的标签数据转换为向量

    # 获取图像数据
    file_img = open(file)
    line = file_img.readline()

    img_data = ''
    while line:
        line = line.replace('\n', '')  # 去掉字符串换行符
        img_data = img_data + line  # 将每个文件里读取的数据拼接成一个字符串，方便转为向量
        line = file_img.readline()

    # 将字符串列表转换为浮点数列表
    img_data = map(float, img_data)
    img_data = list(img_data)

    return np.array(img_data)


if __name__=="__main__":

   X,Y = getData("../data/trainingDigits")
   print(X.shape)
   print(Y.shape)