# Binary Input Linear Model: Verification of Computed Results
import torch

# 输入数据
x = torch.tensor([1.0])  # 输入 x
y = torch.tensor([2.0])  # 真实值 y

# 初始化权重和偏置
W1 = torch.tensor([1.0], requires_grad=True)  # 权重 W1
W2 = torch.tensor([2.0], requires_grad=True)  # 权重 W2
b = torch.tensor([1.0], requires_grad=True)   # 偏置 b

# 前向传播
y_pred = W1 * x ** 2 + W2 * x + b  # 预测值
loss = (y_pred - y) ** 2  # 损失函数

# 计算损失值
print("Predicted y: ", y_pred.item())
print("Loss: ", loss.item())

# 反向传播
loss.backward()

# 打印梯度
print(f"Gradient w.r.t W1: {W1.grad.item()}")
print(f"Gradient w.r.t W2: {W2.grad.item()}")
print(f"Gradient w.r.t b: {b.grad.item()}")

