import numpy as np
import BP as bp
import warnings
import LoadData as rd
import cv2 as cv
warnings.filterwarnings("ignore")

w1 = bp.loadModel("./model/w1.txt")
w2 = bp.loadModel("./model/w2.txt")

vec_img = bp.getImageData("../data/images/4_1.png")

#vec_img = rd.getImg("../data/testDigits/1_41.txt")

y = bp.forward(vec_img,w1,w2)

print(np.around(y,5))
print("最大可能")
print(np.argmax(y))

y[np.argmax(y)]= -1

print("次大可能")
print(np.argmax(y))

# a = np.loadtxt("../data/testDigits/5_2.txt")
#
# for i in
#
# cv.imwrite("../data/images/5_4.png",a)