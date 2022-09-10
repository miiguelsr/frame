import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

datos = load_iris()

X = datos.data

y = datos.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 50, test_size = 0.25)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Train accuracy:",accuracy_score(y_true = y_train, y_pred=model.predict(X_train)))
print("Test accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))