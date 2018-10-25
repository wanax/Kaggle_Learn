#######################

# Machine Learning Test 2018/09/16
# Miles Ma
#
########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('2c_weka.csv')

print(data.iloc[0, :])

#print(plt.style.available)
#plt.style.use('ggplot')

##EXPLORATORY DATA ANALYSIS (EDA)
####################################################################################

#print(data.head())

#data['class'] = data['class'].map({"Abnormal":1, "Normal":0})

#print(data.describe())

#color_list = ['red' if i == 1 else 'green' for i in data['class']]

#pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
#                           c = color_list,
#                           figsize = [15,15],
#                           diagonal = 'hist',
#                           alpha = 0.5,
#                           s = 200,
#                           marker = '*',
#                           edgecolor = 'black')
#plt.show()

#sns.countplot(x='class', data=data)

##K-NEAREST NEIGHBORS (KNN)
####################################################################################

#x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
#
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(x_train, y_train)
#pred = knn.predict(x_test)
#print("accuracy: ", knn.score(x_test, y_test))

## Find the best n_neighbors
#**********************************************

#neigs = np.arange(1, 25)
#train_accuracy = []
#test_accuracy = []
#
#for k in neigs:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(x_train, y_train)
#    train_accuracy.append(knn.score(x_train, y_train))
#    test_accuracy.append(knn.score(x_test, y_test))
#    
#plt.figure(figsize = [13, 8])
#plt.plot(neigs, test_accuracy, label='test')
#plt.plot(neigs, train_accuracy, label='train')
#plt.legend()

##REGRESSION
####################################################################################

#data1 = data[data['class'] == 'Abnormal']
#
#x = data1.pelvic_incidence
#x = np.array(data1.pelvic_incidence).reshape(-1, 1)
#
#y = data1.sacral_slope
#y = np.array(y).reshape(-1, 1)

#plt.figure(figsize=[10, 10])
#plt.scatter(x = x1, y = y1)

##CROSS VALIDATION
####################################################################################

#reg = LinearRegression()
#k = 5
#cv_result = cross_val_score(reg, x, y, cv = k)
#print("CV score: ", cv_result)
#print("mean: ", np.sum(cv_result)/k)

##HYPERPARAMETER TUNING
####################################################################################

#data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
#x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
#
#grid = {'n_neighbors': np.arange(1,50)}
#knn = KNeighborsClassifier()
#knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
#knn_cv.fit(x,y)# Fit
#
## Print hyperparameter
#print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
#print("Best score: {}".format(knn_cv.best_score_))

print(np.logspace(-3, 3, 7))






































