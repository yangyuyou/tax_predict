import base64
import cv2
import numpy as np
from PIL import Image
from predict import predict_test2

def ocr_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = 140#重庆
    img[img > avg] = 255
    img[img <= avg] = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                k = np.sum(img[i - 1:i + 2, j - 1:j + 2] == 0)
                if k < 2:
                    img[i, j] = 255
    img = img[1:-1, 1:-1]
    img=img[:,0:83]
    height = img.shape[0]
    width = img.shape[1]

    from matplotlib import pyplot as plt

    plt.plot(img.mean(axis=0))
    # plt.show()
    # 设置最小的文字像素高度
    min_val = 15
    start_i = -1
    end_i = -1
    number = 0
    # 列分割
    bs = []
    for i in range(width):
        # 有字符
        #  img=np.array(img)

        if (not img[:, i].all()):
            end_i = i
            if (start_i < 0):
                start_i = i
                pass
            elif (end_i - start_i >= 25):
                # point.append((start_i,end_i))
                img1 = img[:, start_i:end_i]
                number += 1
                img1 = Image.fromarray(img1)
                # img1.show()
                a = predict_test2(img1)
                b = a.tolist()
                bs.append(b[0])
                # img1.save(root + str(number)+'.jpg')
                start_i, end_i = -1, -1
                pass
            else:
                pass

        # 无字符
        elif (img[:, i].all() and start_i >= 0):
            if (end_i - start_i >= min_val):
                img1 = img[:, start_i:end_i]
                number += 1
                img1 = Image.fromarray(img1)
                # img1.show()
                a = predict_test2(img1)
                b = a.tolist()
                bs.append(b[0])
                start_i, end_i = -1, -1
                pass
            else:
                pass

    # label=''
    # for i in bs:
    #     label=label+str(i)
    return bs#返回的是验证码所有字符

def compute(bs):
     number={'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
     if len(bs)==3:
         if bs[0] in number.keys() and bs[2] in number.keys():
             if bs[1] == '加':
                 value = number[bs[0]] + number[bs[2]]
                 return value
             elif bs[1] == '减':
                 value = number[bs[0]] - number[bs[2]]
                 return value
             elif bs[1] == '乘':
                 value = number[bs[0]] * number[bs[2]]
                 return value
             else:
                 value = int(number[bs[0]] / number[bs[2]])
                 return value
     else:
         return ''

def read_img_chongqin(b):
    img_str=base64.b64decode(b)
    img_Arr=np.frombuffer(img_str,np.uint8)
    img=cv2.imdecode(img_Arr,cv2.IMREAD_COLOR)
    label=ocr_img(img)
    result=compute(label)
    return result

# with open('img/2.png', 'rb') as f:
#     label = f.read()
#     img_b64 = base64.b64encode(label)
#     a=img_b64.decode('utf-8')
#     res=read_img_chongqin(a)
#     print(res)
# for p in paths:
#     p1 = p.replace('.jpg', '.txt')
#     result=read_img_chongqin(p)
#     with open(p1,'w',encoding='utf-8') as f:
#         f.write(str(result))


# # 测准确率
# from train_chongqin import testP, testSet
#
# n=0
# for i in range(len(testP)):
#     img,label=testSet[i]
#     # img.show()
#     a = predict_test2(img)
#     b=a.tolist()
#     if b[0]==label:
#         n+=1
#
# c=n/len(testP)
#
# print('准确率：',c)



# paths=glob('chongqin/chongqin2/*.jpg')
# sum=0
# for p in paths:
#     p1=p.replace('.jpg','.txt')
#     img=cv2.imread(p)
#     bs=ocr_img(img)
#     with open(p1,'r',encoding='utf-8') as f:
#         label1=f.read()
#     if len(bs)==3:
#         if bs[0]==label1[0] and bs[1]==label1[1] and bs[2]==label1[2]:
#             sum=sum+1
# acc=sum/1000
# print(acc)















