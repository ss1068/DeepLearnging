import torch
import numpy as np
import matplotlib.pyplot as plt
#准备数据集
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0.0],[0.0],[1.0]])
#设计模型
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
model = LogisticRegression()
#损失和优化器
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch',epoch,'loss',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w',model.linear.weight.item())
print('b',model.linear.bias.item())
x = np.linspace(0, 10, 200)
x_data = torch.Tensor(x).view(200, 1)
y_pred = model(x_data)
y = y_pred.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5])
plt.show()
