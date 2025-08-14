import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
# y = w * x + b
def model(x):
    return w * x + b
def loss(x,y):
    y_hat = model(x)
    return (y_hat - y)**2
w_range = np.arange(0, 3.1,0.1)
b_range = np.arange(0, 3.1,0.1)
print(w_range)
MSE_list = []
for w in np.arange(0,3.1,0.1):
    MSE_sublist = []
    for b in np.arange(0,3.1,0.1):
        loss_sum = 0
        for x,y in zip(x_data,y_data):
            loss_sum += loss(x,y)
        MSE = loss_sum / len(x_data)
        MSE_sublist.append(MSE)
    MSE_list.append(MSE_sublist)
#绘图
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection='3d')
X,Y = np.meshgrid(w_range,b_range)
Z = np.array(MSE_list)
surf = ax.plot_surface(X, Y, Z.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()