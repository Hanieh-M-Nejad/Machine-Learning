import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""**Data**"""

df = pd.read_csv('2-dataset.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
plt.scatter(X,y)
plt.show()

"""**Dataset split**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

"""**Linear Regression**"""

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

"""**Predicted y vs Real y**"""

y_hat = model.predict(X_test)
print(y_hat)
print(y_test)