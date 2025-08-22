import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
train_df = pd.read_csv('./dataset/OGPC/train.csv')
test_df = pd.read_csv('./dataset/OGPC/test.csv')

X_train_df = train_df.drop(columns=['id', 'target'])
y_train_series = train_df['target']

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_series)

class OGPCDataset(Dataset):
    def __init__(self, features_df, labels_arr=None, is_test=False):
        self.is_test = is_test
        self.x_data = torch.tensor(features_df.values, dtype=torch.float32)
        if not self.is_test:
            self.y_data = torch.tensor(labels_arr, dtype=torch.long)
        self.len = len(self.x_data)
    def __getitem__(self, index):
        if self.is_test:
            return self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
train_dataset = OGPCDataset(features_df=X_train_df, labels_arr=y_train_encoded, is_test=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 87)
        self.l2 = torch.nn.Linear(87, 62)
        self.l3 = torch.nn.Linear(62, 48)
        self.l4 = torch.nn.Linear(48, 24)
        self.l5 = torch.nn.Linear(24, 9)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx,(inputs,target) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
def test():
    print("\n--- 开始在测试集上进行预测 ---")
    X_test_df = test_df.drop(columns=['id'])
    test_dataset = OGPCDataset(features_df=X_test_df,is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())
    final_predictions = np.concatenate(predictions, axis=0)
    submission_df = pd.DataFrame(final_predictions, columns=le.classes_)
    submission_df.insert(0, 'id', test_df['id'])
    submission_df.to_csv('submission.csv', index=False)
    print("提交文件 'submission.csv' 已成功生成！")
    print(submission_df.head())


if __name__ == '__main__':
    num_epochs = 10  # 设定训练周期数
    for epoch in range(num_epochs):
        train(epoch)

    # 训练结束后，在测试集上预测并生成文件
    test()






