#预测
import cv2
import numpy as np
from PIL import Image
from sklearn.externals import joblib

def predict_test1(img):
    img = img.convert('L')
    img = img.resize((14, 36))
    imgs=[]
    imgs.append(img)
    for i in range(len(imgs)):
       img=np.array(imgs[i])
       img=img.ravel()
       imgs[i]=img
    img_test=np.array(imgs)
    # img_x=X_test[index]
    # img=img_x.reshape((36,16))
    # img=Image.fromarray(img)
    # img.show()
    # x = X_test[index].reshape((1, len(X_test[index])))
    Ssvm2 = joblib.load('Ssvm2.model')
    y_predict = Ssvm2.predict(img_test)
    return y_predict

# p='shanghai/shanghai8/0/1.jpg'
# img=Image.open(p)
# a=predict_test1(img)
# print(a)

def predict_test2(img):
    # img = img.convert('L')
    img=img.resize((14,36))
    imgs=[]
    imgs.append(img)
    for i in range(len(imgs)):
       img=np.array(imgs[i])
       img=img.ravel()
       imgs[i]=img
    img_test=np.array(imgs)
    # img_x=X_test[index]
    # img=img_x.reshape((36,16))
    # img=Image.fromarray(img)
    # img.show()
    # x = X_test[index].reshape((1, len(X_test[index])))
    best_Csvm = joblib.load('best_Csvm.model')
    y_predict = best_Csvm.predict(img_test)
    return y_predict

# import yanzhengma
#
# yanzhengma.ulr_address('重庆','chongqin/chongqin5/',5)

# p='chongqin/chongqin1/177/2.jpg'
# img=cv2.imread(p)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# avg = 140#重庆
# img[img > avg] = 255
# img[img <= avg] = 0
# img = img[1:-1, 1:-1]
# img=Image.fromarray(img)
# img.show()
# a=predict_test2(img)
# print(a)
