import base64
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image

from img_from import read_p
from predict import predict_test1
from train_shanghai import testP, testSet, X_test, Y_test

#测单张图片
# def ocr_img(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # avg = 140#重庆
#     avg = 80  # 上海
#     # print(avg)
#     img[img > avg] = 255
#     img[img <= avg] = 0
#     img = img[1:-1, 1:-1]
#     height = img.shape[0]
#     width = img.shape[1]
#     img2=Image.fromarray(img)
#     # img2.show()
#     from matplotlib import pyplot as plt
#
#     plt.plot(img.mean(axis=0))
#     # plt.show()
#     # 设置最小的文字像素高度
#     min_val = 8
#     start_i = -1
#     end_i = -1
#     number = 0
#     # 列分割
#     bs = []
#     for i in range(width):
#         # 有字符
#         #  img=np.array(img)
#
#         if (not img[:, i].all()):
#             end_i = i
#             if (start_i < 0):
#                 start_i = i
#                 pass
#             elif (end_i - start_i >= 25):
#                 # point.append((start_i,end_i))
#                 img1 = img[:, start_i:end_i]
#                 number += 1
#                 img1 = Image.fromarray(img1)
#                 # img1.show()
#                 a = predict_test1(img1)
#                 b = a.tolist()
#                 bs.append(b[0])
#                 start_i, end_i = -1, -1
#                 pass
#             else:
#                 pass
#
#         # 无字符
#         elif (img[:, i].all() and start_i >= 0):
#             # if i==146:
#             #     img1 = img[:, start_i:end_i]
#             #     number += 1
#             #     basename = os.path.basename(p)
#             #     root = os.path.join('tep', 'chongqin1', basename.replace('.jpg', '/'))
#             #     if not os.path.exists(root):
#             #         os.makedirs(root)
#             #     img1 = Image.fromarray(img1)
#             #     img1.save(root + str(number) + '.jpg')
#             #     pass
#             if (end_i - start_i >= min_val):
#                 img1 = img[:, start_i:end_i]
#                 number += 1
#                 img1 = Image.fromarray(img1)
#                 # img1.show()
#                 a = predict_test1(img1)
#                 b = a.tolist()
#                 bs.append(b[0])
#                 start_i, end_i = -1, -1
#                 pass
#             else:
#                 pass
#     label=''
#     for i in bs:
#         label=label+str(i)
#
#
#     # with open(p, 'w', encoding='utf-8') as f:
#     #     for i in bs:
#     #         f.write(i)
#     return label.strip('')  #返回的是验证码所有字符
#
# def read_img_shanghai(b):
#     img_str = base64.b64decode(b)
#     img_Arr = np.frombuffer(img_str, np.uint8)
#     img = cv2.imdecode(img_Arr, cv2.IMREAD_COLOR)
#     label=ocr_img(img)
#     return label
# img=cv2.imread('img/2.jpg')
# label=ocr_img(img)
# print(label)
# 测单个分割图片
# img=cv2.imread('shanghai/shanghai4/a/1.jpg')
# img=Image.fromarray(img)
# # img.show()
# a = predict_test1(img)
# b=a.tolist()
# print('预测结果：', b[0])

# import sys
#
# sys.path.append('')
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

charactersS = ' 0123456789abcdefghijklmnopqrstuvwxyz  '
model = load_model('shanghai.h5')
def predict(image):
    """
    crnn模型，ocr识别
    """
    w, h = image.size
    image = (np.array(image.convert('L')) / 255.0 - 0.5) / 0.5

    image = image.reshape((1, h, w, 1))
    out = ''
    if model is not None:
        y_pred = model.predict(image)
        out = decode(y_pred[:, 2:, :], charactersS)  ##
    return out


def decode(pred, charactersS):
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(charactersS)
    delStr = [n - 1, ]
    for i in range(length):
        if t[i] not in delStr and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return ''.join(char_list)[:4]

import six
def ocr_from_bsting(string):
    b = base64.b64decode(string)
    buf = six.BytesIO()
    buf.write(b)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    res=predict(img)
    return res
# def read_img_shanghai(b):
#     img_str = base64.b64decode(b)
#     img_Arr = np.frombuffer(img_str, np.uint8)
#     img=ocr_from_bsting(img_Arr)
#     # img = cv2.imdecode(img_Arr, cv2.IMREAD_COLOR)
#     # img=Image.fromarray(img)
#     # img=img.convert('RGB')
#     label=predict(img)
#     return label
# # 测准确率
# n=0
# for i in range(len(testP)):
#     img,label=testSet[i]
#     # img.show()
#     a = predict_test1(img)
#     b=a.tolist()
#     if b[0]==label:
#         n+=1
#
# c=n/len(testP)
#
# print('准确率：',c)

# from sklearn.externals import joblib
# Ssvm1=joblib.load('Ssvm1.model')
# Y_pred=Ssvm1.predict(X_test)
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(Y_test,Y_pred)
# print(acc)


# paths=glob('shanghai/shanghai7/shanghai/*.jpg')
# sum=0
# for p in paths:
#     p1=p.replace('.jpg','.txt')
#     img=cv2.imread(p)
#     label=ocr_img(img)
#     with open(p1,'r',encoding='utf-8') as f:
#         label1=f.read()
#     if label==label1:
#         sum=sum+1
# acc=sum/1000
# print(acc)

