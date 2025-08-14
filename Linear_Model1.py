import numpy as np
import matplotlib.pyplot as plt



#输入x,y
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#模型 y = w * x
def model(x_data):
    return x_data*w

#损失
def loss(x,y):
    yhat = model(x)
    return (yhat - y)**2
#穷举
MSE_list = []
w_list = []
for w in np.arange(0.0,4.1,0.1) :
    # print(w)
    #计算损失
    loss_sum = 0
    for x,y in zip(x_data,y_data):
        loss_sum += loss(x,y)
    # print("loss_sum:",loss_sum)
    #求MSE
    MSE = loss_sum / len(x_data)
    MSE_list.append(MSE)
    w_list.append(w)
#画图
plt.plot(w_list,MSE_list)
plt.ylabel('MSE')
plt.xlabel('w')
plt.show()
