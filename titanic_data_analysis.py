import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")

print(train_df.head())
print(test_df.head())

train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

print(train_df.head())
print(test_df.head())

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Encode Sex column
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})


X = train_df.drop('Survived', axis=1).to_numpy(dtype=np.float64)
y = train_df['Survived'].values.reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

print(X_mean)
print(X_std)

X = (X - X_mean)/X_std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_loggistic_regression(X, y, lr=0.01, epochs=1000):
    m,n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        dw = (1/m)*np.dot(X.T, (y_pred - y))
        db = (1/m)*np.sum(y_pred - y)

        weights -=lr * dw
        bias -= lr * db
    return weights, bias

weights, bias = train_loggistic_regression(X, y, lr=0.01, epochs=2000)

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)

y_pred = predict(X, weights, bias)

accuracy = np.mean(y_pred == y)
print("Accuracy: ", accuracy)

conf_matrix = pd.crosstab(
    y.flatten(),
    y_pred.flatten(),
    rownames=['Actual'],
    colnames=['Predicted']
)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.show()