'''
    Author: Nikita Koloskov (IPT Ukraine
    
    This programs makes the model, that predicts
    the result of tossing the yut stick
    depending on the external parameters.
    
    Parameters that could affect the result:
        - height
        - surface softness
        
        
    Outputs:
        0 - fallen on flat side
        1 - fallen on round side
        
    Classification is made using SVM method, as it
    shows good performance even on small datasets.
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('data_general.csv')
labels = data['result']
features = data[['length','radius','theta','height','surface','spin']]

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)

print('Classification accuracy: %.2f' % acc)
print('Coefficients: ')
print(' Length: %.2f' % clf.coef_[0][0])
print(' Radius: %.2f' % clf.coef_[0][1])
print(' Theta: %.2f' % clf.coef_[0][2])
print(' Height: %.2f' % clf.coef_[0][3])
print(' Surface: %.2f' % clf.coef_[0][4])
print(' Spin: %.2f' % clf.coef_[0][5])
print('Bias: %.2f' % clf.intercept_)
