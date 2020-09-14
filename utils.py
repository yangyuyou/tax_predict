from glob import glob

import cv2
import numpy as np
from PIL import Image


def resize_img(img, label):
    imgW, imgH = img.size
    scale = imgH * 1.0 / 32
    w = imgW / scale
    w = int(w)
    H = max(32, imgH)
    mean = int(np.array(img).mean())  # 求图像均值
    if w < 512 and np.random.randint(1, 10000) > 9950:
        ##合并

        imgsize = img.resize((w, 32), Image.BILINEAR)  # 缩放成（w,32）的双线性图像
        tmp = np.zeros((H, int((imgW + w) * 1.1)), dtype='uint8')#产生一个36行，24列的零矩阵
        tmp[:] = mean
        if np.random.randint(0, 10) > 5:
            tmp[:imgH, :imgW] = np.array(img)
            wbia = np.random.randint(imgW, tmp.shape[1] - imgsize.size[0])
            hbia = 0 if tmp.shape[0] - 32 <= 0 else np.random.randint(0, tmp.shape[0] - 32)
            tmp[hbia:hbia + 32, wbia:wbia + w] = np.array(imgsize)
        else:
            hbia = 0 if tmp.shape[0] - 32 <= 0 else np.random.randint(0, tmp.shape[0] - 32)
            tmp[hbia:hbia + 32, :w] = np.array(imgsize)
            wbia = np.random.randint(w + 1, tmp.shape[1] - imgW)
            tmp[:imgH, wbia:wbia + imgW] = np.array(img)
        img = Image.fromarray(tmp)
        label = label * 2

    if np.random.randint(0, 10000) < 10:
        degree = np.random.uniform(-4, 4)
        img = img.rotate(degree, center=(img.size[0] / 2, img.size[1] / 2), expand=1, fillcolor=mean)
    return img, label

def addImage(img1, img2, alpha):  # 图片叠加
    h, w = img1.shape[:2]  # 图片的高度和宽度
    """
        函数要求两张图必须是同一个size
        alpha，beta，gamma可调
    """
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)  # 图片缩放，interpolation属于插入方式

    beta = 1 - alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)  # 实现图片的叠加
    return img_add

def img_filter(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # Sobel算子计算图像梯度
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absx = cv2.convertScaleAbs(x)  # 转回uint8
    absy = cv2.convertScaleAbs(y)
    dist = cv2.addWeighted(absx, 0.5, absy, 0.5,
                           0)  # 由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来。0.5代表x和y方向的权重
    return dist


def blur_argument(img):
    img = np.array(img)
    dist_img = img_filter(img)
    rand = np.random.uniform(0, 3)  # 生成一个随机数rand,0<=rand<3
    if rand > 1:
        rand = 1
    IMG_Add = addImage(np.array(img), dist_img, rand)
    return Image.fromarray(IMG_Add)



def aug_img(img):
    if np.random.randint(0, 1000) > 10:
        img = blur_argument(img)
        return img
    img = np.array(img)
    kernels = [cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),  # 矩形
               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),  # 椭圆形
               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 交叉形

               ]  # cv2.getStructuringElement( ) 返回指定形状和尺寸的结构元素（核）。
    kernel = kernels[np.random.randint(0, 3)]
    rand = np.random.randint(0, 100)
    if rand > 30:
        dilation = cv2.dilate(img, kernel)  # 膨胀
    elif rand > 60:
        dilation = cv2.erode(img, kernel)  # 腐蚀
    else:
        dilation = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 进行闭运算，先进行膨胀操作，再进行腐蚀操作。
    img = Image.fromarray(dilation)
    img = blur_argument(img)
    return img


def add_noise(img):
    ##在某一通道增加矩形区域干扰
    img = img.convert('RGB')
    img = np.array(img)
    ymin = np.random.randint(0, 32)
    h = np.random.randint(0, 10)
    ymax = min(ymin + h, 32)
    img[ymin:ymax, :, 0] = np.random.randint(0, 255)
    return Image.fromarray(img).convert('L')


class Dataset1():

    def __init__(self, paths):
        self.nsamples = len(paths)
        self.paths = paths

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        # paths=glob('25/*.jpg')
        p = self.paths[index]
        img = Image.open(p)
        p1 = p.replace('.jpg', '.txt')
        with open(p1, 'r', encoding='utf-8') as f:
            label = f.read()
        #         a=p.split('/')[1].replace('.jpg','')
        #         label=a.split('-')[4]
        img = img.convert('L')
        if img.size[0] > 200:
            size = img.size[0] + np.random.randint(-50, 100), img.size[1] + np.random.randint(-5, 5)
            ##图像压缩或拉长几个像素
            img = img.resize(size)
            w = img.size[0]
            if w > 2048 or w < 10:
                return self[index + 1]

            if label == '###' or len(label) > 50:
                return self[index + 1]

            img, label = resize_img(img, label)
            img = aug_img(img)  # 进行图像膨胀和腐蚀
            if np.random.randint(0, 1000) > 990:
                img = add_noise(img)  # 在某一区域增加矩形干扰

        return (img, label)



def generator1(dataset=None):

    N = len(dataset)
    index = [i for i in range(N)]
    np.random.shuffle(index)  # 打乱数据集
    k = 0
    imgs = []
    labels = []
    j = 0
    while j < N:

        ind = index[k]
        k += 1
        im, label = dataset[ind]
        if im.size[0] > 200 and len(label) > 6 and np.random.randint(0, 1000) > 800:
            im = im.resize((im.size[0] + np.random.randint(-100, 200), im.size[1]), Image.LINEAR)

        im=im.resize((14,36))#重庆
        imgs.append(im)
        labels.append(label.replace(' ', ''))
        j += 1
    # imgs = alignCollate(32, 100, True)(imgs)  # alignCollate返回的不是np.array(images),改代码
    for i in range(len(imgs)):
        img = np.array(imgs[i])
        img = img.ravel()
        imgs[i] = img
    X = np.array(imgs)
    Y = np.array(labels)

    return X, Y

# p='shanghai/shanghai5/0/1.jpg'
# img=cv2.imread(p)
# img1=Image.open(p)
# # img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
# img1 = img1.convert('L')
# im=np.array(img1)
# print(img)
# print(im)
