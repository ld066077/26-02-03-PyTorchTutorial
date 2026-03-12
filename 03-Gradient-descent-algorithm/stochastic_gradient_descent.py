x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 权重的初始值

# 前向传播函数：预测 y = x * w
def forward(x):
    return x * w

# 损失函数：计算 MSE（均方误差）
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 梯度计算函数：d_loss/d_w = 2 * x * (x * w - y)
def gradient(x, y):
    return 2 * x * (x * w - y)

# 训练前预测
print('Predict (before training)', 4, forward(4))

# 训练循环
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)      # 1. 计算梯度
        w = w - 0.01 * grad        # 2. 更新权重（学习率为 0.01）
        print("\tgrad: ", x, y, grad)
        l = loss(x, y)             # 3. 计算损失
    
    print("progress:", epoch, "w=", w, "loss=", l)

# 训练后预测
print('Predict (after training)', 4, forward(4))
