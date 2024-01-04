# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\NIT\JANUARY\3 JAN(logistic regression pra)\2nd,3rd\2.LOGISTIC REGRESSION CODE\logit classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#for this observation let me selcted as 100 observaion for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

'''
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
#we mentioned feature scaling only to independent variable not dependent variable at all


#Next step is we are going to build the logistic model and appy this model into our dataset 
#This is linear model library thats why we called from sklear.linear_model

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#now we will use the confusion matrix to evalute

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr




bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance


#build first classification model in python '''
##########################################################################





