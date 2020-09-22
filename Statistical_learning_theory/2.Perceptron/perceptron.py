import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class PerceptronModel(object):

    def __init__(self,iter=1000):
        self.iter = iter
        self.b = 0
        self.learning_rate = 0.1
    
    def sign(self,x,w,b):
        y = np.dot(x,w)+b
        return y

    def perceptron(self, X_train, y_train):
        # 重み
        self.w = np.ones(len(X_train[0]), dtype=np.float32)
        for i in range(self.iter):
            for j in range(len(X_train)):
                X = X_train[j]
                y = y_train[j]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.learning_rate * np.dot(y, X)
                    self.b = self.b + self.learning_rate * y
                    # print(f'Round {i}:{self.iter} training')



def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    print(df)
    data = np.array(df.iloc[:100, [0, 1, -1]])

    return data


def main():
    data= load_data()
    X, y = data[:,:-1], data[:,-1]
    y = np.array([1 if i == 1 else -1 for i in y])
    perceptron = PerceptronModel()
    perceptron.perceptron(X, y)

    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()





