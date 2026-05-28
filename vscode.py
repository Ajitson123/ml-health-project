import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
v = pd.read_csv("diabetes.csv")
print(v.head())
print(v.shape)
x = v.drop("Outcome", axis= 1)
y = v['Outcome']
scaler = StandardScaler()
x = scaler.fit_transform(x)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(x)
v['Cluster'] = clusters
print("\nCluster vs Actual Outcome:")
print(pd.crosstab(v['Cluster'], v['Outcome']))
plt.figure(figsize=(10, 6))
plt.scatter(v[v['Cluster'] == 0]['Glucose'], v[v['Cluster'] == 0]['BMI'], c = "blue", label='Cluster 0', alpha=0.5)
plt.scatter(v[v['Cluster'] == 1]['Glucose'], v[v['Cluster'] == 1]['BMI'], c = "orange", label='Cluster 1', alpha=0.5)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title("KMeans Cluster - Glucose vs BMI")
plt.legend()
plt.show()
print("\nCluster summaries:")
print(v.groupby('Cluster')[['Glucose', 'BMI', 'Age']].mean())