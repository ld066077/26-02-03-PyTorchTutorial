from pathlib import Path
import numpy as np
import torch

torch.manual_seed(0)

data_path = Path(__file__).with_name("test.csv")
xy = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]]) #,[-1]确保y_data计算为矩阵不是向量(,-1)；x_data和y_data必须都为矩阵


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch:4d}, loss={loss.item():.6f}")

with torch.no_grad():
    pred = model(x_data)
    pred_label = (pred >= 0.5).float()

print("\ntrain data:")
print(x_data)
print("\nlabel:")
print(y_data.squeeze())
print("\nprediction probability:")
print(pred.squeeze())
print("\nprediction label:")
print(pred_label.squeeze())
