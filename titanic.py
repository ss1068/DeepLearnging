import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset



class MyDataset(Dataset):
    def __init__(self,filepath,is_test=False):
        self.is_test = is_test
        df = pd.read_csv(filepath)
        df['Age'].fillna(df['Age'].mean(), inplace=True)


        # 1女 0 男
        df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
        feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']
        label_column = 'Survived'
        #num 数组
        x_np = df[feature_columns].values
        #转为Tensor
        self.x_data = torch.tensor(x_np, dtype=torch.float32)
        #不是测试集，才加载标签
        if not self.is_test:
            label_column = 'Survived'
            y_np = df[label_column].values
            self.y_data = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
        else:
            self.passenger_ids = df['PassengerId'].values
        self.len = len(x_np)

    def __getitem__(self, index):
        if self.is_test:
            return self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
dataset = MyDataset("./train.csv")
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(torch.nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x
model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
if __name__ == '__main__':
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("\n--- 开始在测试集上进行预测 ---")
    test_dataset = MyDataset("./test.csv",is_test=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=2)
    model.eval()
    predictions = []
    passenger_ids = test_dataset.passenger_ids
    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs)
            y_pred = (outputs > 0.5).int()
            predictions.extend(y_pred.squeeze().tolist())
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    submission_df.to_csv('submission.csv', index=False)