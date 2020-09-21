#Author:XYZ
#Date:2020-09-20

#　正則化

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
REGULARIZATION = 0.0001

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

# 残差　L1正則化
def residuals_L1(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * REGULARIZATION * abs(p)))
    return ret


# 残差　L2正則化
def residuals_L2(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * REGULARIZATION * np.square(p)))
    return ret


def main():
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)

    # 目標モデルの出力にホワイトノイズを加える
    y_ = real_func(x)
    y = [temp + np.random.normal(0, 0.1) for temp in y_]

    #N为多項式の次数
    N = 9
    # 多项式パラメータを初期化
    p_init = np.random.rand(N + 1)
    # 最小二乗法
    p_lsq = leastsq(residuals, p_init, args=(x, y))

    p_lsq_L1  = leastsq(residuals_L1, p_init, args=(x, y))
    p_lsq_L2  = leastsq(residuals_L2, p_init, args=(x, y))
    print('Fitting Parameters without regularizer:', p_lsq[0])
    print('Fitting Parameters with L1:', p_lsq_L1[0])
    print('Fitting Parameters with L2:', p_lsq_L2[0])

    #可視化
    plt.plot(x_points, real_func(x_points), label='real',color='black')

    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='without regularizer',color='red')
    plt.plot(x_points,fit_func(p_lsq_L1[0], x_points),label='L1')
    plt.plot(x_points,fit_func(p_lsq_L2[0], x_points),label='L2')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
