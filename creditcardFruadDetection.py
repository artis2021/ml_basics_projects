# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# print("Path to dataset files:", path)

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("creditcard.csv")
print(data.head())

print(data.info())
print(data["Class"].value_counts())

sns.countplot(x = "Class", data=data)
plt.title("Fraud Vs Normal Transactions")
plt.show()

scaler = StandardScaler()
data["Amount"] = scaler.fit_transform(data[["Amount"]])

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=20,
    random_state=42
)

model.fit(X_train, y_train)

y_read = model.predict(X_test)

print("\nClassification Report: ")
print(classification_report(y_test, y_read))

print("\nConfusion Matrix: ")
cm = confusion_matrix(y_test, y_read)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()