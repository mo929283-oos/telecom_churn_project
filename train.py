


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

"""# loading & cheking data"""

df=pd.read_csv(r"C:\Users\Admin\Desktop\telecom_churn_project\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()

df.shape

df.info()

"""# removing Unnecessary fetures"""

df=df.drop(["customerID"], axis=1) # also we can write df=df.drop(columns=['customerID'])
df.head()

"""# Exploring Unique values categorical columns"""

fetures = ["tenure","MonthlyCharges","TotalCharges"]
for col in df.columns :
  if col not in fetures :
    print(col,df[col].unique())

"""# checking for missing values"""

df.isnull().sum()

df["TotalCharges"].values

"""## Python sees this value as a string (text), not a number!"""

df[df["TotalCharges"] == " "]

len(df[df["TotalCharges"] == " "])

df["TotalCharges"] = df["TotalCharges"].replace(" ", 0.0)

df["TotalCharges"] = df["TotalCharges"].astype(float) # Convert column data type from string (object) to float (numeric)

df.info()

"""# Visualizing churn distribution"""

sns.countplot(data=df, x="Churn",hue="Churn") #hue means when group change color change
plt.show()

df.describe()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Churn Distribution by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Customer Count")
plt.show()

plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, fill=True, palette="Set1")
plt.title("Tenure Distribution: Churned vs. Retained Customers")
plt.xlabel("Tenure (Months)")
plt.show()

plt.figure(figsize=(9, 5))
sns.kdeplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    fill=True,
    common_norm=False,
    palette="Set1"
)

plt.title("Monthly Charges Distribution: Churned vs. Retained Customers", fontsize=14)
plt.xlabel("Monthly Charges ($)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Set3")
plt.title("Monthly Charges Impact on Churn")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="InternetService", hue="Churn", palette="Set2")
plt.title("Churn Rate by Internet Service Type")
plt.xlabel("Internet Service")
plt.ylabel("Customer Count")
plt.show()

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn",
    palette="Set2"
)
plt.title("Churn Rate by Payment Method", fontsize=14)
plt.xlabel("Payment Method", fontsize=12)
plt.ylabel("Customer Count", fontsize=12)
plt.xticks(rotation=15) # because not mix texts


plt.show()

# heatmap

"""# cleaning categorical columns for consistency"""

# Replacing 'No phone service' --> No
# Replacing 'No internet service' --> No
df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

"""# Encoding binary(yes\no) columns to binary"""

yes_no_columns = ["Partner","Dependents","PhoneService","MultipleLines",'OnlineSecurity',"OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in yes_no_columns :
  df[col].replace({'Yes':1 ,'No': 0},inplace = True)

df.head()

"""# Encoding multi-categorical varibles Using One-Hot Encoding"""

df2 = pd.get_dummies(data=df, columns=["InternetService", "Contract", "PaymentMethod"], dtype=int)
df2.head()

df2['gender'] = df2['gender'].map({'Male': 1, 'Female': 0})

df2.info()

X = df2.drop(columns=['Churn'])
y = df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


X_train.head()

smote =SMOTE(random_state=42)
X_train_smote,y_train_smote =smote.fit_resample(X_train,y_train)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_smote, y_train_smote)

y_pred_logistic = logistic_model.predict(X_test)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)

print("Logistic Regression Results")
print("=" * 40)
print("Accuracy :", accuracy_logistic)
print("Precision:", precision_logistic)
print("Recall   :", recall_logistic)
print("F1 Score :", f1_logistic)

print(classification_report(y_test, y_pred_logistic))

from sklearn.metrics import roc_curve, roc_auc_score

y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)

roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})",
    linewidth=2
)

plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

cm_logistic = confusion_matrix(y_test, y_pred_logistic)

print("Confusion Matrix:")
print(cm_logistic)

ConfusionMatrixDisplay(
    confusion_matrix=cm_logistic,
    display_labels=logistic_model.classes_
).plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.show()

logistic_results = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy_logistic,
    "Precision": precision_logistic,
    "Recall": recall_logistic,
    "F1 Score": f1_logistic
}

logistic_results

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# Prediction probabilities
y_prob_dt = dt_model.predict_proba(X_test)[:,1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1 Score :", f1_score(y_test, y_pred_dt))
print("ROC AUC  :", roc_auc_score(y_test, y_prob_dt))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_dt))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Churn','Churn'],
    yticklabels=['No Churn','Churn']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_dt)

roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Decision Tree (AUC = {roc_auc_dt:.2f})",
    linewidth=2
)

plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

importance = dt_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.show()

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)

plt.show()

sample = X_test.iloc[[0]]

prediction = dt_model.predict(sample)[0]
probability = dt_model.predict_proba(sample)[0][1]

print("Prediction:", "Churn" if prediction==1 else "No Churn")
print("Probability:", round(probability*100,2),"%")

print(importance_df.head(10))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(X_test)[:,1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))
print("ROC AUC  :", roc_auc_score(y_test, y_prob_rf))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Churn","Churn"],
    yticklabels=["No Churn","Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)

roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc_rf:.2f})",
    linewidth=2
)

plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

importance = pd.DataFrame({

    "Feature":X.columns,
    "Importance":rf_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features (Random Forest)")

plt.show()

sample = X_test.iloc[[0]]

prediction = rf_model.predict(sample)[0]

probability = rf_model.predict_proba(sample)[0][1]

print("Prediction :", "Churn" if prediction==1 else "No Churn")

print("Probability :", round(probability*100,2),"%")

print(importance.head(10))

risk = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_rf,
    "Churn Probability": y_prob_rf
})

risk = risk.sort_values(
    by="Churn Probability",
    ascending=False
)

risk.head(10)

risk["Risk Level"] = pd.cut(
    risk["Churn Probability"],
    bins=[0,0.4,0.7,1],
    labels=["Low","Medium","High"]
)

risk.head(10)

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train_smote, y_train_smote)

y_pred_xgb = xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

from sklearn.metrics import roc_curve, roc_auc_score

# Get probabilities
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)

# Calculate ROC-AUC
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Plot
plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc_xgb:.2f})",
    linewidth=2
)

# Random Classifier
plt.plot(
    [0, 1],
    [0, 1],
    'r--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - XGBoost")

plt.legend()

plt.show()

import matplotlib.pyplot as plt
import pandas as pd

feature_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))

feature_importance.tail(10).plot(kind="barh")

plt.title("Top 10 Features - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dictionary to store all models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Store results
results = []

for name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

# Create DataFrame
results_df = pd.DataFrame(results)

# Sort models by F1-Score
results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

# Display results
print("Model Comparison:")
print(results_df)

# Plot model comparison

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)

plt.show()

# Get the best model based on F1-Score

best_model_name = results_df.loc[0, "Model"]
best_f1_score = results_df.loc[0, "F1-Score"]

print("Best Model:", best_model_name)
print("Best F1-Score:", round(best_f1_score, 4))

print("Best Accuracy Model:")
print(results_df.loc[results_df["Accuracy"].idxmax(), ["Model", "Accuracy"]])

print("\nBest Precision Model:")
print(results_df.loc[results_df["Precision"].idxmax(), ["Model", "Precision"]])

print("\nBest Recall Model:")
print(results_df.loc[results_df["Recall"].idxmax(), ["Model", "Recall"]])

print("\nBest F1-Score Model:")
print(results_df.loc[results_df["F1-Score"].idxmax(), ["Model", "F1-Score"]])

print("\nBest ROC-AUC Model:")
print(results_df.loc[results_df["ROC-AUC"].idxmax(), ["Model", "ROC-AUC"]])


import joblib


joblib.dump(xgb_model, 'xgb_model.pkl')


joblib.dump(scaler, 'scaler.pkl')


joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')


print("- xgb_model.pkl")
print("- scaler.pkl")
print("- model_columns.pkl")