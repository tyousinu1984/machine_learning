#Author:XYZ
#Date:2020-09-23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

class K_Nearest_Neighbors:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 個点
        parameter: p-ノルム
        """
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):

        knn_list = []
        for i in range(self.n_neighbors):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.n_neighbors, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 統計
        knn = [k[-1] for k in knn_list]

        count = Counter(knn)

        max_count = sorted(count.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def validation(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

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

    clf = K_Nearest_Neighbors(X_train, y_train)
    
    accurate_rate = round(clf.validation(X_test, y_test),3)*100

    print(f"accurate_rate is {accurate_rate}%")
    test_point_predict = clf.predict(test_point)
    
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