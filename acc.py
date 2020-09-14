#重庆
#画学习曲线
from glob import glob

import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import  matplotlib.pyplot as plt


#输入数据
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils import Dataset1, generator1

paths = glob('shanghai/shanghai9/*/*.jpg')#上海
# paths=glob('chongqin/chongqin4/*/*.jpg')#重庆

dataSet = Dataset1(paths)
X, y= generator1(dataSet)

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1.0,5)):
    plt.title(title)#图像标题
    if ylim is not None: #y轴限制不为空时
        plt.ylim(*ylim)
    plt.xlabel("Training examples") #两个标题
    plt.ylabel('score')
    train_sizes1,train_scores1,test_scores1=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean1=np.mean(train_scores1,axis=1)
    train_scores_std1=np.std(train_scores1)
    test_scores_mean1 = np.mean(test_scores1, axis=1)
    test_scores_std1 = np.std(test_scores1, axis=1)
    # train_sizes2, train_scores2, test_scores2 = learning_curve(estimator[1], X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
    # test_scores_mean2 = np.mean(test_scores2, axis=1)
    # test_scores_std2 = np.std(test_scores2, axis=1)
    # train_sizes3, train_scores3, test_scores3 = learning_curve(estimator[2], X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
    # test_scores_mean3 = np.mean(test_scores3, axis=1)
    # test_scores_std3 = np.std(test_scores3, axis=1)
    # train_sizes4, train_scores4, test_scores4 = learning_curve(estimator[3], X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
    # test_scores_mean4 = np.mean(test_scores4, axis=1)
    # test_scores_std4 = np.std(test_scores4, axis=1)
    # train_sizes5, train_scores5, test_scores5 = learning_curve(estimator[4], X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
    # test_scores_mean5 = np.mean(test_scores5, axis=1)
    # test_scores_std5 = np.std(test_scores5, axis=1)
    # train_sizes6, train_scores6, test_scores6 = learning_curve(estimator[5], X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
    # test_scores_mean6 = np.mean(test_scores6, axis=1)
    # test_scores_std6 = np.std(test_scores6, axis=1)
    plt.grid()#背景设置为网格线
    plt.fill_between(train_sizes1,train_scores_mean1,-train_scores_std1,train_scores_mean1+train_scores_std1,alpha=0.1,color='g')#函数回把模型准确性的平均值的上下方差的空间用颜色填充
    plt.fill_between(train_sizes1,test_scores_mean1-test_scores_std1,test_scores_mean1+test_scores_std1,alpha=0.1,color='r')
    # plt.fill_between(train_sizes2, test_scores_mean2 - test_scores_std2, test_scores_mean2 + test_scores_std2, alpha=0.1,color='g')
    # plt.fill_between(train_sizes3, test_scores_mean3 - test_scores_std3, test_scores_mean3 + test_scores_std3, alpha=0.1, color='orange')
    # plt.fill_between(train_sizes4, test_scores_mean4 - test_scores_std4, test_scores_mean4 + test_scores_std4,alpha=0.1, color='purple')
    # plt.fill_between(train_sizes5, test_scores_mean5 - test_scores_std5, test_scores_mean5 + test_scores_std5,alpha=0.1, color='orange')
    # plt.fill_between(train_sizes6, test_scores_mean6 - test_scores_std6, test_scores_mean6 + test_scores_std6,alpha=0.1, color='purple')
    plt.plot(train_sizes1,train_scores_mean1,'o-',color='g',label='Training Score')
    #然后用plt.plot()函数画出模型准确性的平均值
    plt.plot(train_sizes1,test_scores_mean1,'o-',color='r',label='Test Score')
    # plt.plot(train_sizes2, test_scores_mean2, 'o-', color='g', label='MultinomialNB')
    # plt.plot(train_sizes2, test_scores_mean2, 'o-', color='g', label='OneVsOneClassifier')
    # plt.plot(train_sizes4, test_scores_mean4, 'o-', color='black', label='RandomForestClassifier')
    # plt.plot(train_sizes3, test_scores_mean3, 'o-', color='orange', label='OneVsRestClassifier')
    # plt.plot(train_sizes4, test_scores_mean4, 'o-', color='purple', label='OneVsOneClassifier')
    plt.legend(loc='best') #显示图例
    plt.show()
    # train_scores_mean=np.mean(train_scores,axis=1)
    # train_scores_std=np.std(train_scores)

# a = KNeighborsClassifier(n_neighbors=3)
# b = MultinomialNB(alpha=1.0)
# c = DecisionTreeClassifier(max_depth=12)
# d = RandomForestClassifier(max_depth=5)
e = OneVsRestClassifier(SVC(kernel='linear',probability=True,random_state=0))
# f = OneVsOneClassifier(SVC(kernel='linear',probability=True,random_state=0))
cv=ShuffleSplit(n_splits=7,test_size=0.1,random_state=0)
plot_learning_curve(e,'Learning Curve',X,y,ylim=(0.4,1.01),cv=cv)
# plot_learning_curve(SMnb,'Learning Curve',X,y,ylim=(0.5,1.01),cv=cv)
#取重庆数据300，用SVM一对多或一对一都行
