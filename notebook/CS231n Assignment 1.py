import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

%matplotlib inline

digits = load_digits()

print(digits.target.shape)
print(digits.data.shape)

plt.imshow(digits.images[4])

W = np.random.normal(0, 1, [10, 64])
b = np.random.normal(0, 1, [10, 1])
x = digits.data.reshape([1797, 64, 1])
target = digits.target.reshape(1797, 1)
y = target.reshape([-1, 1])
learning_rate = 0.01

#y_hat 계산
def return_y_hat(weight=None, data=None, bias=None):
    y_hat = weight.dot(data) + bias
    return y_hat

#sigmoid 계산
def sigmoid(y, y_hat):
    difference = y_hat - y
    cost = 1 / (1+np.exp(-difference))
    return cost

#Gradient
def gradient_descent(cost, W, lr=None):
    diff_sig = cost * (1 - cost)
    W = W - (lr * diff_sig)
    return W

#트레이닝
plot_x = []
plot_y = []

for j in range(1, 10000):
    for i in range(1, 1000):
        y_hat = return_y_hat(weight=W, data=x[i], bias=b)
        cost = sigmoid(y[i], y_hat)
        W = gradient_descent(cost, W, lr=0.01)
    if j % 100 == 0:
        print("{}th training completed".format(j))
        plot_x.append(j)
        plot_y.append(np.sum(cost))

plt.plot(plot_x, plot_y)
plt.show()

#싸이킷런
from sklearn import svm
x = digits.data.reshape([1797, 64])
target = digits.target.reshape(1797, 1)
y = target.reshape([-1, 1])

x_train, x_test = x[:1500], x[1500: -1]
y_train, y_test = y[:1500], y[1500: -1]

clf = svm.SVC()
clf.fit(x_train, y_train)

clf.predict(x_test)

print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))