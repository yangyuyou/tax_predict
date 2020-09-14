from utils import *
from glob import glob
from sklearn.model_selection import train_test_split
# from key import alphabetChinese  ##需要训练识别的字符集

#输入数据
paths = glob('shanghai/shanghai9/*/*.jpg')
trainP, testP = train_test_split(paths, test_size=0.1)# 用于将矩阵随机划分为训练子集和测试子集
trainSet = Dataset1(trainP)
testSet = Dataset1(testP)

#产生训练集和测试集
X_train, Y_train = generator1(trainSet)
X_test, Y_test = generator1(testSet)

#knn开始训练
# from sklearn.neighbors import KNeighborsClassifier
# Sknn1 = KNeighborsClassifier(n_neighbors=3)
# Sknn1.fit(X_train, Y_train)

#multinomialNB(多项式朴素贝叶斯算法)
# from sklearn.naive_bayes import MultinomialNB
# SMnb=MultinomialNB(alpha=1.0)
# SMnb.fit(X_train,Y_train)

#决策树分类算法
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import cross_val_score
# import matplotlib.pyplot as plt
# Stree=DecisionTreeClassifier(max_depth=13)
# Stree.fit(X_train,Y_train)



#随机森林算法
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import cross_val_score
# import matplotlib.pyplot as plt
# SRtree=RandomForestClassifier(max_depth=13)
# SRtree.fit(X_train,Y_train)


#SVM算法一对多
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# Ssvm3=OneVsRestClassifier(SVC(kernel='linear',probability=True,random_state=0))
# Ssvm3.fit(X_train,Y_train)

#SVM算法一对一
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import SVC
# Ssvm3=OneVsOneClassifier(SVC(kernel='linear',probability=True,random_state=0))
# Ssvm3.fit(X_train,Y_train)

#保存模型
from sklearn.externals import joblib
# joblib.dump(Ssvm3,'Ssvm3.model')
# SKnn=joblib.load('SKnn.model')

#准确率
# Y_pred=Cknn.predict(X_test)
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(Y_test,Y_pred)
# print(acc)


#交叉验证
# Ssvm1=joblib.load('Sknn1.model')
# from sklearn.cross_validation import cross_val_score
# a=cross_val_score(Ssvm3,X_test,Y_test,cv=8,scoring='accuracy')
# print(a.mean())








# Y_predict=knn.predict(X_test)
# from sklearn.metrics import accuracy_score
# accuracy_score(Y_predict,Y_test)
# print(accuracy_score())#正确率