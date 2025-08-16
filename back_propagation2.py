import torch
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w1 = torch.tensor([0.1])
w1.requires_grad_(True)
w2 = torch.tensor([0.0])
w2.requires_grad_(True)
b = torch.tensor([0.0])
b.requires_grad_(True)
def forward(x):
    return w1 * (x ** 2) + w2 * x  + b
def loss(x,y):
    y_hat = forward(x)
    return (y_hat - y) ** 2
for epoch in range(70000):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        # print('\tgrad:', x, y, w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data -= w1.grad.data * 0.002
        w2.data -= w2.grad.data * 0.002
        b.data -= b.grad.data * 0.002
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    # print("progress:", epoch, l.item())
print(w1.item(),w2.item(),b.item())