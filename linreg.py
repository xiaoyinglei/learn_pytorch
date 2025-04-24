import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l  # 动手学深度学习的工具包

# 真实权重和偏差
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成带有噪声的模拟数据 (1000条样本，2个特征)
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 加载数据迭代器函数
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 把 features 和 labels 打包到一起
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回 DataLoader

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))  # 取出一个小批量样本用于查看结构（可删）

# 创建线性模型：输入是2维，输出是1维
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))  # 创建包含一个全连接层的神经网络

# 初始化权重参数
net[0].weight.data.normal_(0, 0.01)  # 用均值为0、标准差为0.01的正态分布初始化权重
net[0].bias.data.fill_(0)           # 偏置初始化为0

# 定义损失函数：均方误差 MSE
loss = nn.MSELoss()

# 定义优化器：随机梯度下降（SGD），学习率为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练循环
num_epochs = 3
for epoch in range(num_epochs):  # 总共训练3轮
    for X, y in data_iter:       # 遍历每个小批量
        l = loss(net(X), y)      # 计算预测值和真实值之间的损失
        trainer.zero_grad()      # 梯度清零（每次迭代都需要清零）
        l.backward()             # 反向传播，计算梯度
        trainer.step()           # 执行一步梯度下降，更新参数
    l = loss(net(features), labels)  # 每轮结束后在所有数据上计算损失
    print(f'epoch {epoch + 1}, loss {l:f}')  # 打印损失

# 查看学习到的参数与真实参数之间的误差
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
