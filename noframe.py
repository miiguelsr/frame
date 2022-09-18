import numpy as np
import pandas as pd
import requests
import io

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def logistic_regression(X, Y, learning_rate, iterations):
  m = X.shape[1]
  n = X.shape[0]

  W = np.zeros((n,1))
  B = 0

  cost_list = []

  for i in range(iterations):

    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    dW = (1/m)*np.dot(A-Y, X.T)
    dB = (1/m)*np.sum(A-Y)

    W -= learning_rate*dW.T
    B -= learning_rate*dB

    cost_list.append(cost)
  return W, B, cost_list

columns = ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
url = "https://raw.githubusercontent.com/miiguelsr/frame/659b263663a68bd9ebef6b142c67f6af61be0e6a/wine.data" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')), names=columns)
df

from sklearn.model_selection import train_test_split

X = df[["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]].to_numpy()
y = df["Class"].factorize()[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=567)

X_train = X_train.T
y_train = y_train.reshape(1, X_train.shape[1])
X_test = X_test.T
y_test = y_test.reshape(1, X_test.shape[1])

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

iterations = 100000
learning_rate = 0.0005
W, B, cost_list = logistic_regression(X_train, y_train, learning_rate=learning_rate, iterations=iterations)