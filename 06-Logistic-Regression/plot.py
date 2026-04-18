import torch
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 0. 创建模型
model = LogisticRegressionModel()

# 2. 加载训练好的参数
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 3. 生成要画图的输入点
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float32).view(200, 1)

# 4. 用模型预测
with torch.no_grad():
    y_t = model(x_t)

y = y_t.numpy()

# 5. 画图
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Logistic Regression")
plt.plot([0, 10], [0.5, 0.5], "r--", label="Threshold = 0.5")
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression Prediction Curve")
plt.grid()
plt.legend()

# 6. 保存图片
plt.savefig("logistic_curve.png", dpi=300, bbox_inches="tight")
print("图片已保存为 logistic_curve.png")

