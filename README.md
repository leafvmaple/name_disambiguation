# 同名消歧-竞赛

该思路分为以下几步：

1. 从论文pub中提取出title, keywords, venue, org等features。
2. 将得到的featrues使用word2vec构建embedding。
3. 对于每一篇论文, 使用相同作者的其他论文作为正对象，其他作者的论文作为负对象，送入triplet网络中训练。
4. 对于每一个作者，构建论文矩阵，将论文feature相似度满足一定阈值的连边，送入GAE网络中训练。
5. Bidiretional LSTM作为Encoder，FullyConnected Layer作为Decoder进行聚类。

其中2, 3, 4步分别对应Emb, Global和Local。
