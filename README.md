# Personal-Recommendation-Action-
一些经典的个性化推荐算法的实现，从理论推导到实战
本项目实现了一些经典的个性化推荐召回算法，包括*基于邻域的推荐算法（LFM）*、*基于图的推荐算法（Personal Rank）*、*基于深度学习的推荐算法（Item2vec）*。其次本项目通过两种常用的排序模型，*逻辑回归（LR）和 GBDT*，进行样本的选择和处理举例，并且实现了*GBDT和LR混合模型*。
## 数据集 ##
[MovieLens](https://movielens.org/)</br>
[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income)

## 个性化召回算法--LFM（Latent Factor Model）
1. 公式推导
- LFM建模公式
```math
p(u,i)=p_u^Tq_i=\sum_{f=1}^Fp_(u_f)q_(i_f)
```
