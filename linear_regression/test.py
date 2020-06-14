from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import LinearRegression
# X, y = datasets.load_diabetes(True)
# print(dataset.DESCR)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


model = LinearRegression(learning_rate = 1.0)
mse = model.fit(X_train, y_train, 400, True)
predictions = model.predict(X_test)
mse = np.array(mse)
y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
# plt.plot(mse)
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()