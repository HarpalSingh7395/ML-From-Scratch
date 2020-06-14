from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression
from matplotlib import pyplot as plt
dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
print(X_train.dtype)
print(X_train.shape)

regression = LogisticRegression()
accuracy = regression.fit(X_train, Y_train, 1600, 0.01, True)

predictions = regression.predict(X_test)
data = [True if predictions[i] == Y_test[i] else False for i in range(len(X_test))]
print(data)
plt.figure()
plt.plot([i for i in range(len(accuracy))], accuracy)
plt.show()


