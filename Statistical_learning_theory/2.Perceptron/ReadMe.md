## 定義
1 パーセプトロンは，二項分類器である。分類器への入力特徴ベクトル$x$、重みベクトルが$\omega $である時、出力は次のようになる　$$ f(x)=sign(w\cdot x +b) $$　パーセプトロンモデルは、入力空間（又は　特徴空間　feature space）の分離超平面$ w\cdot x +b =0 $に対応する

2 パーセプトロンの学習則は　$$\underset{w ,b}{min}L(w,b) = -\sum_{x_{i}\in M}y_{i}(w\cdot x_{i}+b)$$ 損失関数は、超平面から誤分類点の距離の総計である。


## アルゴリズム
確率的勾配降下法　Stochastic Gradient Descent
$$\begin{matrix}
w = w+ \eta y_{i}x_{i}\\ 
b = b+ \eta y_{i}\;\;\;\;\;
\end{matrix}$$

インスタンス点が誤分類された場合、つまり分離超平面の反対側にある場合は、$ w $、$ b $の値を調整して、誤分類された点が正しく分類されるまで、分離ハイパープレーンを未分類ポイントの側に移動する。

## 実装
### アルゴリズムによるパーセプトロン

![](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/perceptron.png)

[アルゴリズムによるパーセプトロン](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/perceptron.py)

### sklearnによるパーセプトロン

![](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/perceptron_by_sklearn.png)

[アルゴリズムによるパーセプトロン](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/perceptron_by_sklearn.py)



### アルゴリズムによる交差検証

![](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/holdout_validation.png)

[アルゴリズムによる交差検証](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/holdout_validation.py)

accurate rate is 0.9333333333333333

### sklearnによる交差検証

![](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/holdout_validation_by_sklearn.png)

[アルゴリズムによる交差検証](https://github.com/tyousinu1984/machine_learning/blob/master/Statistical_learning_theory/2.Perceptron/holdout_validation_by_sklearn.py)

accurate rate is 1.0

おそらく、sklearnなにかの裏技がある。
