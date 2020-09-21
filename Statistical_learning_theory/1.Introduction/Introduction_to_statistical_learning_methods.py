#Author:XYZ
#Date:2020-09-20

#　最小二乗法のモデル推定

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目標モデル
def real_func(x):
    return np.sin(2*np.pi*x)

# 推定モデル 多項式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals(p, x, y):
    ret = fit_func(p, x) - y
    return ret


def main():
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)

    # 目標モデルの出力にホワイトノイズを加える
    y_ = real_func(x)
    y = [temp + np.random.normal(0, 0.1) for temp in y_]

    # #N为多項式の次数
    # N = 9
    # # 多项式パラメータを初期化
    # p_init = np.random.rand(N + 1)
    # # 最小二乗法
    # p_lsq = leastsq(residuals, p_init, args=(x, y))
    # print('Fitting Parameters:', p_lsq[0])

    # #可視化
    # plt.plot(x_points, real_func(x_points), label='real')
    # plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    # plt.plot(x, y, 'bo', label='noise')
    # plt.legend()
    # plt.show()
    # N为多項式の次数
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for N in range(1, 10):
        ax = fig.add_subplot(3, 3, N)
        p_init = np.random.rand(N + 1)
        p_lsq = leastsq(residuals, p_init, args=(x, y))
        print(f'N= {N},Fitting Parameters:', p_lsq[0])

        ax.plot(x_points, real_func(x_points), label='real')
        ax.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
        ax.plot(x, y, 'bo', label='noise')
        ax.set_title(f'N= {N}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()