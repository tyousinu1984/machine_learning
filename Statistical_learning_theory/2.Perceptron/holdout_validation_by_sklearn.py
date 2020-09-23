#Author:XYZ
#Date:2020-09-23

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # print(df)
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data

def holdout_validation_data_set(data):
    X, y = data[:,:-1], data[:,-1]
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, y_train, X_test, y_test

def model_test(X_test,y_test,w,b):
    error_count = 0

    for d in range(len(X_test)):
        X = X_test[d]
        y = y_test[d]
        result = y * np.dot(X, w) + b
        if result <= 0:
            error_count += 1

    accurate_rate = 1 -(error_count/len(X_test))
    return accurate_rate


def main():
    data= load_data()
    X_train, X_test, y_train, y_test = holdout_validation_data_set(data)

    clf = Perceptron(fit_intercept=False,max_iter=1000,tol=None, shuffle=False)
    clf.fit(X_train, y_train)
    print(clf.coef_, clf.intercept_)


    accurate_rate = model_test(X_test,y_test,clf.coef_[0],clf.intercept_)
    print(f"accurate rate is {accurate_rate}")

    x_points = np.linspace(4, 7, 10)
    y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()





