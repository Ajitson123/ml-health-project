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
v = pd.read_csv("diabetes.csv")
print(v.head())
print(v.shape)
x = v.drop("Outcome", axis= 1)
y = v['Outcome']
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
v_f = RandomForestClassifier(n_estimators=100, random_state=42)
v_f.fit(x_train, y_train )
print("Random forest(100 tress): ")
print("train accuracy: ", accuracy_score(y_train, v_f.predict(x_train)))
print("test accuracy: ", accuracy_score(y_test, v_f.predict(x_test)))
v_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
v_oob.fit(x_train, y_train)
print("\noob score: ", v_oob.oob_score)
cross_score = cross_val_score(v_f, x, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print("\nCross-validation scores: ", cross_score)
print("Mean cross-validation score: ", np.mean(cross_score))
print("standard deviation of cross-validation scores: ", np.std(cross_score))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
skf_scores = cross_val_score(v_f, x, y, cv=skf, scoring='accuracy')
print("\nStratifiedKFold (10 splits) cross-validation scores: ", skf_scores)
print("Mean StratifiedKFold score: ", skf_scores.mean())
print("Standard deviation of StratifiedKFold scores: ", skf_scores.std())
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
ada.fit(x_train, y_train)
print("ada boost test accuracy: ", accuracy_score(y_test, ada.predict(x_test)))
print("\nFeature importance:")
for feature, importances in zip(x.columns, v_f.feature_importances_):
    print(f"{feature}: {importances:.4f}")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(x_train, y_train)
print("gradient boosting test accuracy: ", accuracy_score(y_test, gb.predict(x_test)))
gb_cv = cross_val_score(gb, x, y, cv=10, scoring="accuracy")
print("gb cv mean accuracy:", gb_cv.mean())
print("gb cv standard deviation:", gb_cv.std())
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(x_train, y_train)
print("XGBoost test accuracy: ", accuracy_score(y_test, xgb.predict(x_test)))
xgb_cv = cross_val_score(xgb, x, y, cv=10, scoring="accuracy")
print("XGBoost cv mean accuracy:", xgb_cv.mean())
print("XGBoost cv standard deviation:", xgb_cv.std())
print("\nFeature importance:")
for feature, importances in zip(x.columns, xgb.feature_importances_):
    print(f"{feature}: {importances:.4f}")
plt.figure(figsize=(10, 6))
plt.barh(x.columns, v_f.feature_importances_)
plt.xlabel("Importance")
plt.title("Random forest feature importance - pima diabetes")
plt.tight_layout()
plt.show()
print("\n ----comparison----")
print("desicion tree(max_depth=3) test accuracy: 0.7597")
print("random forest (100 trees) test accuracy:", accuracy_score(y_test, v_f.predict(x_test)))