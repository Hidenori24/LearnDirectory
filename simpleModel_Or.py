#!usr/bin/env python
# coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

#モデル定義
model = L.Linear(1,1)
optimizer = optimizers.SGD()
optimizer.setup(model)

# 学習回数
times = 50
# 入力ベクトル
x = Variable(np.array([[1]], dtype=np.float32))

# 正解ベクトル
t = Variable(np.array([[2]], dtype=np.float32))

# 学習ループ
for i in range(0,times):
    # 勾配初期化
    optimizer.zero_grads()

    # モデル予測
    y = model(x)

    # モデルが出した答えの表示
    print(y.data)

    # 損失計算
    loss = F.mean_squared_error(y,t)

    # 逆伝播
    loss.backward()

    # optimizerを更新する
    optimizer.update()
