import matplotlib.pyplot as plt
import LoadData as rd
import OPTBP as opt
import numpy as np

if __name__ =="__main__":
    x_train, y_train = rd.getData("../data/trainingDigits")

    bp = opt.BP(1934, 1024, 64, 10,1000,x_train, y_train)

    bp.start()
    losses = bp.getLosses()
    bp.saveModel("./model")
    losss = []
    for i in range(0, len(losses), 10):
        losss.append(losses[i])

    print(losss)
    #使用训练集进行测试
    x_test, y_test = rd.getData("../data/testDigits")
    right_n = 0
    all_n = len(x_test)
    for i in range(len(x_test)):

        Y_ = bp.predictSmaple(x_test[i])
        if np.argmax(Y_) == np.argmax(y_test[i]):
            right_n = right_n + 1

    print("总数---" + str(all_n))
    print("正确个数" + str(right_n))
    print("准确率----")
    print(float(right_n / all_n))



    plt.plot(losss)
    plt.show()