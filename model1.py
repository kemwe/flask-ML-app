import pandas as pd
import numpy as np
#import pickle
data=pd.read_csv('C:/Users/user/Desktop/titanic data/train.csv')
data.head()

data.drop(['PassengerId', 'Pclass','Name', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin',],axis=1,inplace=True)

data["Age"].fillna(value=data["Age"].mean(),inplace=True)
data["Embarked"].fillna(value='c',inplace=True)

df=pd.get_dummies(data[["Sex","Embarked"]])

data=pd.concat([data,df],axis=1)
data.drop(["Embarked","Sex"],axis=1,inplace=True)

from sklearn.model_selection import train_test_split 
x=data.drop("Survived",axis=1)
y=data['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

# from sklearn.externals import joblib
# #import joblib
# joblib.dump(lr,'model.pkl')
# lm=joblib.load('model.pkl')
# model_columns = list(x.columns)
# joblib.dump(model_columns, 'model_columns2.pkl')
#print("models's columns dumped")
import pickle
pickle.dump(lr, open("model2.pkl","wb"))

model_columns = list(x.columns)
pickle.dump(model_columns,open("model_columns.pkl","wb"))
print("models's columns dumped")