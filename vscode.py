import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
v = pd.read_csv('diabetes.csv')
x = v.drop('Outcome', axis=1)
y = v['Outcome']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
svm_linear = SVC(kernel='linear', C=1, random_state=42)
svm_linear.fit(x_train_scaled, y_train)
print("SVM Linear Accuracy:", accuracy_score(y_test, svm_linear.predict(x_test_scaled)))
svm_rbf = SVC(kernel='rbf', C=1, random_state=42)
svm_rbf.fit(x_train_scaled, y_train)
print("SVM RBF Accuracy:", accuracy_score(y_test, svm_rbf.predict(x_test_scaled)))
cv_scores = cross_val_score(svm_rbf, scaler.fit_transform(x), y, cv=10, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
