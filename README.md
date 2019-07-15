# Personal-Recommendation-Action-
一些经典的个性化推荐算法的实现，从理论推导到实战
本项目实现了一些经典的个性化推荐召回算法，包括*基于邻域的推荐算法（LFM）*、*基于图的推荐算法（Personal Rank）*、*基于深度学习的推荐算法（Item2vec）*。其次本项目通过两种常用的排序模型，*逻辑回归（LR）和 GBDT*，进行样本的选择和处理举例，并且实现了*GBDT和LR混合模型*。
## 数据集 ##
[MovieLens](https://movielens.org/)</br>
[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income)

1. 公式推导

- LFM建模公式
```math
p(u,i)=p_u^Tq_i=\sum_{f=1}^Fp_(u_f)q_(i_f)
```
其中，F是隐向量的维度。
- 损失函数
```math
L=\sum_{(u,i)\in D}(p(u,i)-p^{LFM}(u,i))^2+\sigma|p_u|^2+\sigma|q_i|^2
```
后两项正则化系数是使得模型简单化，防止过拟合。
- 算法求导
```math
\frac{\partial L}{\partial p_{u_f}}=-2((p(u,i))-p^{LFM}(u,i))q_{i_f}+2\partial{p_{u_f}} 
```
```math
\frac{\partial L}{\partial q_{i_f}}=-2((p(u,i))-p^{LFM}(u,i))p_{u_f}+2\partial{q_{i_f}}
```
-迭代更新
```math
p_{u_f}=p_{u_f}-\beta \frac{\partial L}{\partial p_{u_f}}
```
```math
q_{i_f}=q_{i_f}-\beta \frac{\partial L}{\partial q_{i_f}}
```

2. 负样本选取
