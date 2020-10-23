
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

class DataSetSpliterator(object):
    def __init__(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        
        self.data = np.array(df.iloc[:100, : ])

    def holdout_data_set(self, test_size=0.3):
        X, y = self.data[:,:-1], self.data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test



class GaussianNaiveBayes:
    def __init__(self):
        self.model = None

    # 期待値
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 偏差値
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 確率密度関数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # X_train処理
    def summarize(self, train_data):
        # 期待値と偏差値を計算
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

        
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }

    # 確率計算
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 分類
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        accurate_rate = right / float(len(X_test))
        return accurate_rate


if __name__ == '__main__':
    test_point = [4.4,  3.2,  1.3,  0.2]

    X_train, X_test, y_train, y_test = DataSetSpliterator().holdout_data_set()

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    test_point_label = model.predict(test_point)
    print(f"The test data belongs to {test_point_label}")
    print("accurate_rate is " + str(model.score(X_test, y_test)))