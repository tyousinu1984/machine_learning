
### 最小二乗法
最小二乗法を使用して曲線を近似する
多項式によって曲線を近似する、多項式の次数N=6から、過学習が確認された。
![avatar](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/1.Introduction/Figure_1.png)
### 正則化

正則化項を使い、過学習を緩和できる。
![avatar](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/1.Introduction/Figure_2.png)

N=9の時、多項式の係数が以下のようだ、

Fitting Parameters without regularizer: [ 4.17184424e+04 -1.87982805e+05  3.55739607e+05 -3.67257215e+05
                                           2.24682278e+05 -8.26652489e+04  1.76612681e+04 -1.99146127e+03
                                          9.51493607e+01 -5.00947715e-02]
                                          
Fitting Parameters with L1: [-1.26433069e+01  3.38541209e-05  1.67007884e+01  2.42401872e+00
                              1.29826711e-03  7.43243952e-04  5.82703979e+00 -2.14922737e+01
                               9.18792951e+00 -4.21286218e-02]
                               
Fitting Parameters with L2: [ -7.190611    -1.22128057   3.59330641   6.61408705   7.03842239
                             3.79839341  -4.46541109 -16.59674552   8.42075839  -0.02260442]
