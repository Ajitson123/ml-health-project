import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
v = pd.read_csv("HeartDiseaseTrain-Test.csv")
# Encoding categorical columns
from sklearn.preprocessing import LabelEncoder
v_encoded = v.copy()
cat_cols = v.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    v_encoded[col] = le.fit_transform(v_encoded[col])
print(v_encoded.head())
print("Encoded shape:", v_encoded.shape)
#feature and target 
x = v_encoded.drop("target", axis=1)
y = v_encoded["target"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_scaled = scaler.fit_transform(x)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric="logloss")
}
results = {}
for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "k-Nearest Neighbors"]:
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        prob_pred = model.predict_proba(x_test_scaled)[:, 1]
        cv = cross_val_score(model, x_scaled, y, cv=10, scoring="accuracy")
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        prob_pred = model.predict_proba(x_test)[:, 1]
        cv = cross_val_score(model, x, y, cv=10, scoring="accuracy")
    results[name] = {
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, prob_pred),
        "CV Accuracy": cv.mean()
    }
print("\n--- MODEL PERFORMANCE ---")
for name, metrics in results.items():
    print(f"{name}: Test Accuracy={metrics['Test Accuracy']:.4f}, ROC AUC={metrics['ROC AUC']:.4f}, CV Accuracy={metrics['CV Accuracy']:.4f}")
names = list(results.keys())
accs = [results[name]["Test Accuracy"] for name in names]
aucs = [results[name]["ROC AUC"] for name in names]
plt.figure(figsize=(12, 5))
x = np.arange(len(names))
plt.bar(x - 0.2, accs, 0.4, label="Test Accuracy")
plt.bar(x + 0.2, aucs, 0.4, label="ROC AUC")
plt.xticks(x, names, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
# ROC curves for all models
plt.figure(figsize=(12, 8))
for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "k-Nearest Neighbors"]:
        prob_pred = model.predict_proba(x_test_scaled)[:, 1]
    else:
        prob_pred = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_pred)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC AUC']:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models- Heart Disease Prediction")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()
