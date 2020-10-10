#Author:XYZ
#Date:2020-09-23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # print(df)
    data = np.array(df.iloc[:, [0, 1, -1]])
    return data


def data_set(data):
    X, y = data[:,:-1], data[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def main(test_point):
    data= load_data()
    X_train, X_test, y_train, y_test = data_set(data)

    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    
    accurate_rate = round(clf_sk.score(X_test, y_test),3)*100
    print(f"accurate_rate is {accurate_rate}%")

    test_point_predict = clf_sk.predict([test_point])[0]
    print(f'Test Point: {test_point_predict}')
    
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.plot(data[100:, 0], data[100:, 1], 'bo', color='green',label='2')
    plt.plot(test_point[0], test_point[1], 'bo',color='red',label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title(f"accurate_rate is {accurate_rate}%,Test Point: {test_point_predict}")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    test_point = [6.0, 3.0]
    main(test_point)