import torch
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w = torch.tensor([1.0])
w.requires_grad_(True)
def forward(x):
    return w * x
def loss(x,y):
    y_hat = forward(x)
    return (y_hat - y) ** 2
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data -= w.grad.data * 0.07
        w.grad.data.zero_()
    print("progress:", epoch, l.item())
print(w.item())