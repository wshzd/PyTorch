
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


# 在模型中必须要定义 forward 函数，backward 函数（用来计算梯度）会被autograd自动创建。 可以在 forward 函数中使用任何针对 Tensor 的操作。
# 
# net.parameters()返回可被学习的参数（权重）列表和值


params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# 测试随机输入32×32。 注：这个网络（LeNet）期望的输入大小是32×32，如果使用MNIST数据集来训练这个网络，请把图片大小重新调整到32×32。

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# 将所有参数的梯度缓存清零，然后进行随机梯度的的反向传播：

net.zero_grad()
out.backward(torch.randn(1, 10))


# Note
# ``torch.nn`` 只支持小批量输入。整个 ``torch.nn`` 包都只支持小批量样本，而不支持单个样本。 例如，``nn.Conv2d`` 接受一个4维的张量， ``每一维分别是sSamples * nChannels * Height * Width（样本数*通道数*高*宽）``。 如果你有单个样本，只需使用 ``input.unsqueeze(0)`` 来添加其它的维数

# 在继续之前，我们回顾一下到目前为止用到的类。
# 
# 回顾:
# 
# torch.Tensor：一个用过自动调用 backward()实现支持自动梯度计算的 多维数组 ， 并且保存关于这个向量的梯度 w.r.t.
# nn.Module：神经网络模块。封装参数、移动到GPU上运行、导出、加载等。
# nn.Parameter：一种变量，当把它赋值给一个Module时，被 自动 地注册为一个参数。
# autograd.Function：实现一个自动求导操作的前向和反向定义，每个变量操作至少创建一个函数节点，每一个Tensor的操作都回创建一个接到创建Tensor和 编码其历史 的函数的Function节点。
# 重点如下：
# 
# 定义一个网络
# 处理输入，调用backword
# 还剩：
# 
# 计算损失
# 更新网络权重
# 损失函数
# 一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
# 
# 译者注：output为网络的输出，target为实际值
# 
# nn包中有很多不同的损失函数。 nn.MSELoss是一个比较简单的损失函数，它计算输出和目标间的均方误差， 例如


output = net(input)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# Now, if you follow loss in the backward direction, using its .grad_fn attribute, you will see a graph of computations that looks like this:
# 
# ::
# 
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# 
# So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Tensors in the graph that has requires_grad=True will have their .grad Tensor accumulated with the gradient.
# 
# For illustration, let us follow a few steps backward:



print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 反向传播
# 调用loss.backward()获得反向传播的误差。
# 
# 但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
# 
# 现在，我们将调用loss.backward()，并查看conv1层的偏差（bias）项在反向传播前后的梯度

net.zero_grad()     # 清除梯度

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)



import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

