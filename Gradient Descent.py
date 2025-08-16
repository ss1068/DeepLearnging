import matplotlib.pyplot as plt
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w = 1.0
def model(x):
    return w * x
def cost():
    cost_sum = 0
    for x,y in zip(x_data,y_data):
        y_hat = model(x)
        cost_sum += (y_hat - y)**2
    return cost_sum / len(x_data)

def gradient():
    loss_sum = 0
    for x,y in zip(x_data,y_data):
        loss_sum += 2 * x * (x * w - y)
    return loss_sum / len(x_data)
cost_list = []
for epoch in range(100):
    cost_val = cost()
    gradient_val = gradient()
    w = w - 0.03 * gradient_val
    cost_list.append(cost_val)
    print("epoch:",epoch,"cost:",cost_val,"gradient:",gradient_val,"w:",w)
#绘图
plt.plot(range(100),cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()