import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x*w+b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        print('w=', w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)

# build unique w and b arrays and meshgrid
w_vals = np.array(sorted(set(w_list)))
b_vals = np.array(sorted(set(b_list)))
W, B = np.meshgrid(w_vals, b_vals)

# reshape mse_list to match meshgrid (loop was w outer, b inner -> shape (len(w), len(b)))
M = np.array(mse_list).reshape(len(w_vals), len(b_vals)).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')

surf = ax.plot_surface(W, B, M, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()