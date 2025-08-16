import matplotlib.pyplot as plt
import random
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w = 1.0
def model(x):
    return w * x
def loss(x,y):
    y_hat = model(x)
    return (y_hat - y ) ** 2

def gradient(x,y):
    return 2 * x * (x * w - y)

w_list = []
for epoch in range(100):
    #在每个epoch开始前，打乱数据顺序
    data = list(zip(x_data,y_data))
    random.shuffle(data)
    for x,y in data:
        loss_val = loss(x,y)
        gradient_val = gradient(x,y)
        w = w - 0.03 * gradient_val
    w_list.append(w)
    print("epoch:",epoch,"w:",w)
#绘图
plt.plot(range(100),w_list)
plt.xlabel("epoch")
plt.ylabel("w")
plt.show()