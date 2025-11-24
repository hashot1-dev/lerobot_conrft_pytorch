# import torch
# A = torch.randn(3, 3)
# torch.geqrf(A)


import torch

# 创建一个张量
x = torch.tensor([4, 0, 1, 2, 1, 2, 3])

# 调用torch.unique并打印结果
out = torch.unique(x)
print(out) # 输出: tensor([0, 1, 2, 3, 4])

# 不进行排序
out = torch.unique(x, sorted=False)
print(out) # 输出: tensor([4, 0, 1, 2, 3])

# 返回原始张量中元素的索引
out, inverse_indices = torch.unique(x, return_inverse=True)
print(out) # 输出: tensor([0, 1, 2, 3, 4])
print(inverse_indices) # 输出: tensor([4, 0, 1, 2, 1, 2, 3])

# 返回每个唯一元素的出现次数
out, counts = torch.unique(x, return_counts=True)
print(out) # 输出: tensor([0, 1, 2, 3, 4])
print(counts) # 输出: tensor([1, 2, 2, 1, 1])

print("!!!!!where")

import torch

# 示例1：形状相同的张量
x = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
y = torch.tensor([[5, 6, 7], [7, 8, 9], [9, 10, 11]])
z = torch.where(x > 5, x, y)
print(z)
# 输出: tensor([[5, 6, 7], [7, 8, 9], [9, 6, 7]])

# 示例2：标量与张量
x = 3
y = torch.tensor([[1, 5, 7]])
z = torch.where(y > 2, y, x)
print(z)
# 输出: tensor([[3, 5, 7]])

# 示例3：形状不同的张量（广播机制）
x = torch.tensor([[1, 3, 5]])
y = torch.tensor([[2], [4], [6]])
z = torch.where(x > 2, x, y)
print(z)
# 输出: tensor([[2, 3, 5], [4, 3, 5], [6, 3, 5]])


print("!!!!!zeros")

# 创建一个2x3的零张量
zero_tensor = torch.zeros(2, 3)
print(zero_tensor)
# 输出：
# tensor([[0., 0., 0.],
# [0., 0., 0.]])

# 创建一个长度为5的一维零张量
zero_tensor = torch.zeros(5)
print(zero_tensor)
# 输出：
# tensor([0., 0., 0., 0., 0.])
print("!!!!!full")
import torch

# 创建一个形状为(2, 3)，所有元素都是2.0的张量
tensor = torch.full((2, 3), 2.0)
print(tensor)

print("!!!!!torch.tensor")

tensorx = torch.tensor((), dtype=torch.int32)
tensorx.new_ones((2, 3))

print("!!!!!torch.cat")
import torch

# 创建两个张量 A 和 B
A = torch.ones(2, 3) # 2x3 的张量
B = 2 * torch.ones(4, 3) # 4x3 的张量

# 按维度 0（行）拼接
C = torch.cat((A, B), 0)
print(C)
# 输出：
# tensor([[1., 1., 1.],
# [1., 1., 1.],
# [2., 2., 2.],
# [2., 2., 2.],
# [2., 2., 2.],
# [2., 2., 2.]])
print(C.size()) # torch.Size([6, 3])

# 创建另一个张量 D
D = 2 * torch.ones(2, 4) # 2x4 的张量

# 按维度 1（列）拼接
C = torch.cat((A, D), 1)
print(C)
# 输出：
# tensor([[1., 1., 1., 2., 2., 2., 2.],
# [1., 1., 1., 2., 2., 2., 2.]])
print(C.size()) # torch.Size([2, 7])

print("!!!!!torch.randperm")
import torch

# 生成一个长度为 10 的随机排列的张量
random_perm = torch.randperm(10)
print(random_perm)

print("!!!!!torch.clamp")

import torch

a = torch.tensor([-1.0, 0.5, 2.0, 3.5])
clamped_a = torch.clamp(a, min=0.0, max=2.0)
print(clamped_a) # 输出: tensor([0.0000, 0.5000, 2.0000, 2.0000])

print("!!!!!torch.multinomial")

import torch
# 定义权重张量
weights = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float)
# 从权重中采样 2 个索引（不放回）
samples = torch.multinomial(weights, 2, replacement=False)
print(samples) # 输出可能为 tensor([2, 1]) 或其他组合
# 从权重中采样 4 个索引（放回）
samples_with_replacement = torch.multinomial(weights, 4, replacement=True)
print(samples_with_replacement) # 输出可能为 tensor([2, 2, 1, 0])

print("!!!!!torch.cdist")
import torch
# 定义两个向量集合
x1 = torch.FloatTensor([[0.1, 0.2, 0], [0.2, 0.3, 0], [0.3, 0.4, 0]])
x2 = torch.FloatTensor([[0.2, 0.3, 0], [0.3, 0.4, 0]])
# 计算两个向量集合之间的距离
distances = torch.cdist(x1, x2)
print(distances)

print("!!!!!torch.stack")

import torch
# 创建两个形状为[3, 3]的张量
T1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
# 使用torch.stack沿着不同的维度合并它们
R0 = torch.stack((T1, T2), dim=0)
R1 = torch.stack((T1, T2), dim=1)
R2 = torch.stack((T1, T2), dim=2)
# 输出合并后的张量及其形状
print("R0:\n", R0)
print("R0.shape:\n", R0.shape)
print("R1:\n", R1)
print("R1.shape:\n", R1.shape)
print("R2:\n", R2)
print("R2.shape:\n", R2.shape)

print("!!!!!torch.exp")

import torch
# 创建一个张量
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
# 对张量的每个元素应用指数函数
y = torch.exp(x)
# 输出结果
print(y) # 输出: tensor([0.1353, 0.3679, 1.0000, 2.7183, 7.3891])
print("!!!!!torch.cumsum")

print("!!!!!torch.nn.Conv2d")
import torch
x1 = torch.arange(0, 6)
print(x1)
# tensor([0, 1, 2, 3, 4, 5])
y1 = torch.cumsum(x1, dim=0)
print(y1)
# tensor([ 0, 1, 3, 6, 10, 15])
y2 = torch.cumsum(x1, dim=-1)
print(y2)
# tensor([ 0, 1, 3, 6, 10, 15])

print("!!!!!torch.nn.Linear")

import torch
import torch.nn as nn
# 创建一个Linear层，输入特征数为20，输出特征数为30
m = nn.Linear(20, 30)
# 创建一个随机输入张量，大小为128x20
input = torch.randn(128, 20)
# 通过Linear层传递输入，得到输出
output = m(input)
# 输出的大小将会是128x30
print(output.size())
print("!!!!!torch.nn.ReLU")
import torch
import torch.nn as nn
# 创建 ReLU 激活函数
relu = nn.ReLU(inplace=True)
# 输入张量
input_tensor = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.0, 2.0]])
# 应用 ReLU 激活函数
output_tensor = relu(input_tensor)
print("输入：", input_tensor)
print("输出：", output_tensor)
print("!!!!!torch.nn.utils.clip_grad_norm_")

import torch
import torch.nn as nn
import torch.nn.utils as utils
# 定义一个简单的线性模型
model = nn.Linear(10, 1)
# 模拟输入和目标
inputs = torch.randn(32, 10)
targets = torch.randn(32, 1)
# 计算损失并反向传播
loss_fn = nn.MSELoss()
loss = loss_fn(model(inputs), targets)
loss.backward()
# 对梯度进行裁剪
max_norm = 1.0
utils.clip_grad_norm_(model.parameters(), max_norm)
print("!!!!!torch.linalg.qr")
import torch
A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
Q, R = torch.linalg.qr(A)
print(Q)
print(R)
print("!!!!!torch.nn.init.orthogonal_")


import torch
import torch.nn as nn
# 创建一个空的张量
w = torch.empty(3, 5)
# 使用orthogonal_函数进行初始化
nn.init.orthogonal_(w)