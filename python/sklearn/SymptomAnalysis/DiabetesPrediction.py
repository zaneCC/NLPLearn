import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix 
# %matplotlib inline

# 导出只有糖尿病患者数据，用于构建后台患者数据
PATH = '/Users/zhouzhan/to_github/NLP/Python/codes/sklearn/SymptomAnalysis/diabetes_data_upload.csv'

dataset = pd.read_csv(PATH)

dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':0})
dataset['class'] = dataset['class'].map({'Positive':1,'Negative':0})
dataset['Polyuria'] = dataset['Polyuria'].map({'Yes':1,'No':0})
dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes':1,'No':0})
dataset['sudden weight loss'] = dataset['sudden weight loss'].map({'Yes':1,'No':0})
dataset['weakness'] = dataset['weakness'].map({'Yes':1,'No':0})
dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes':1,'No':0})
dataset['Genital thrush'] = dataset['Genital thrush'].map({'Yes':1,'No':0})
dataset['visual blurring'] = dataset['visual blurring'].map({'Yes':1,'No':0})
dataset['Itching'] = dataset['Itching'].map({'Yes':1,'No':0})
dataset['Irritability'] = dataset['Irritability'].map({'Yes':1,'No':0})
dataset['delayed healing'] = dataset['delayed healing'].map({'Yes':1,'No':0})
dataset['partial paresis'] = dataset['partial paresis'].map({'Yes':1,'No':0})
dataset['muscle stiffness'] = dataset['muscle stiffness'].map({'Yes':1,'No':0})
dataset['Alopecia'] = dataset['Alopecia'].map({'Yes':1,'No':0})
dataset['Obesity'] = dataset['Obesity'].map({'Yes':1,'No':0})

# 导出只有糖尿病患者数据，用于构建后台患者数据
# dataset.to_excel(TO_PATH)

# 预测模型
corrdata = dataset.corr() # 计算列与列直接的相关系数，返回相关系数矩阵
ax,fig = plt.subplots(figsize=(15,8))
# sns.heatmap(corrdata,annot=True) # 热力图
# sns.distplot(dataset['Age'],bins=30)  
# plt.show()

X1 = dataset.iloc[:,0:-1]
y1 = dataset.iloc[:,-1]

# print(X1.columns)
# print(y1)

print('-------------各个症状与诊断结果进行卡方检验')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

best_feature = SelectKBest(score_func=chi2,k=10)
# 各个症状与诊断结果进行卡方检验
fit = best_feature.fit(X1,y1)
# 得出卡方检验分数
dataset_scores = pd.DataFrame(fit.scores_)
dataset_cols = pd.DataFrame(X1.columns)

featurescores = pd.concat([dataset_cols,dataset_scores],axis=1)
featurescores.columns=['column','scores']
# 输出贡献最大的十个症状
print(featurescores.nlargest(10,'scores'))
# 画出图形显示
featureview=pd.Series(fit.scores_, index=X1.columns)
featureview.plot(kind='barh')

print('-------------通过方差提取特征')
# 通过方差提取特征
from sklearn.feature_selection import VarianceThreshold
feature_high_variance = VarianceThreshold(threshold=(0.5*(1-0.5)))
falls=feature_high_variance.fit(X1)

dataset_scores1 = pd.DataFrame(falls.variances_)
dat1 = pd.DataFrame(X1.columns)

high_variance = pd.concat([dataset_scores1,dat1],axis=1)
high_variance.columns=['variance','cols']
print(high_variance[high_variance['variance']>0.2])

X = dataset[['Polydipsia','sudden weight loss','partial paresis','Irritability','Polyphagia','Age','visual blurring']]
y = dataset['class']

# 切分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 逻辑回归
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,y_train)
# 评价
print('------------- 训练数据的评价')
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=lg, X=X_train ,y=y_train,cv=10)
print("accuracy is {:.2f} %".format(accuracies.mean()*100))
print("std is {:.2f} %".format(accuracies.std()*100))

# 预测数据
print('------------- 预测数据')
pre=lg.predict(X_test)
logistic_regression=accuracy_score(pre,y_test)
print(logistic_regression) # 输出预测数据
print(confusion_matrix(pre,y_test)) # 输出混淆矩阵

# 打印报告
print('------------- 打印报告')
from sklearn.metrics import classification_report
print(classification_report(pre,y_test))