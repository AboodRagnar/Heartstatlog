import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

GetRawData=pd.read_csv('T1/heartstatlog.csv')
GetRawData=GetRawData.replace('?',-99999)

# Features=GetRawData.drop(columns='presence_of _heart _disease')    # old way  of training not sufficent 
# Lable=GetRawData['presence_of _heart _disease']

# bulitClassifier=KNeighborsClassifier(n_neighbors=1) 
# bulitClassifier.fit(Features,Lable)

# prediction_set=[[87,0,4,210,286,1,2,108,1,1.5,2,3,3]]

# print(bulitClassifier.predict(prediction_set))


Features=GetRawData.drop(columns='presence_of _heart _disease')    
Lable=GetRawData['presence_of _heart _disease']

x_train,x_test,y_train,y_test=train_test_split(Features,Lable,test_size=0.3,random_state=4)
x=[]
y=[]
for i in range(1,50):
    bulitClassifier=KNeighborsClassifier(n_neighbors=i) 
    bulitClassifier.fit(x_train,y_train)
    
    prediction=bulitClassifier.predict(x_test)
    
       
    print("hi", bulitClassifier.score(x_test,y_test))
    x.append(metrics.accuracy_score(y_test,prediction))
    y.append(i)

