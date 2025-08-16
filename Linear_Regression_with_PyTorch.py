import torch
#准备数据集
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])
#设计模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()
#损失和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#训练
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch',epoch,'loss',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w',model.linear.weight.item())
print('b',model.linear.bias.item())
x_test_data = torch.tensor([1.0])
y_test_data = model(x_test_data)
print(type(y_test_data))
