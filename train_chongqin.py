from utils import *
from glob import glob
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# from key import alphabetChinese  ##需要训练识别的字符集
import matplotlib.pyplot as plt

#输入数据
paths = glob('chongqin/chongqin4/*/*.jpg')
trainP, testP = train_test_split(paths, test_size=0.1)# 用于将矩阵随机划分为训练子集和测试子集
trainSet = Dataset1(trainP)
testSet = Dataset1(testP)

#产生训练集和测试集
X_train, Y_train= generator1(trainSet)
X_test, Y_test = generator1(testSet)

#当n=3或1时，模型准确率是最高的
# k_range=range(1,10)
# cv_scores=[]
# for n in k_range:
#     Cknn=KNeighborsClassifier(n)
#     scores=cross_val_score(Cknn,X_train,Y_train,cv=10,scoring='accuracy')
#     cv_scores.append(scores.mean())
# plt.plot(k_range,cv_scores)
# plt.xlabel('k')
# plt.ylabel('Accuracy')
# plt.show()


#knn开始训练

# Cknn = KNeighborsClassifier(n_neighbors=3)
# Cknn.fit(X_train, Y_train)


#multinomialNB(多项式朴素贝叶斯算法)
# from sklearn.naive_bayes import MultinomialNB
# CMnb=MultinomialNB(alpha=1.0)
# CMnb.fit(X_train,Y_train)

#决策树分类算法
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import cross_val_score
# import matplotlib.pyplot as plt
# Ctree=DecisionTreeClassifier(max_depth=12)
# Ctree.fit(X_train,Y_train)

#随机森林算法
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import cross_val_score
# CRtree=RandomForestClassifier(max_depth=5)
# CRtree.fit(X_train,Y_train)

#SVM算法一对多
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# Csvm=OneVsRestClassifier(SVC(kernel='linear',probability=True,random_state=0))
# Csvm.fit(X_train,Y_train)


#SVM算法一对一
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import SVC
# best_Csvm=OneVsOneClassifier(SVC(kernel='linear',probability=True,random_state=0))
# best_Csvm.fit(X_train,Y_train)

#保存模型
from sklearn.externals import joblib
# joblib.dump(best_Csvm,'best_Csvm.model')
# # knn=joblib.load('knn.model')


#准确率
# Y_pred=Cknn.predict(X_test)
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(Y_test,Y_pred)
# print(acc)

#交叉验证
best_Csvm=joblib.load('best_Csvm.model')
from sklearn.cross_validation import cross_val_score
a=cross_val_score(best_Csvm,X_test,Y_test,cv=6,scoring='accuracy')
print(a.mean())



# Y_predict=knn.predict(X_test)
# from sklearn.metrics import accuracy_score
# accuracy_score(Y_predict,Y_test)
# print(accuracy_score())#正确率

