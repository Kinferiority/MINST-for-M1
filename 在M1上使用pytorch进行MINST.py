import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms

# 定义超参数
train_batch_size = 64
test_batch_size = 1000
learning_rate =0.01
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epoch =10

# 构建pipeline ，图像处理
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 正则化，降低模型复杂度
])
# 下载数据
train_data = datasets.MNIST("data",train=True,transform=pipeline,download=True)
test_data = datasets.MNIST("data",train=False,transform=pipeline,download=True)

# 加载数据
train_loader = DataLoader(train_data,batch_size=train_batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=test_batch_size,shuffle=False)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=3136,out_features=128),
            nn.Linear(in_features=128,out_features=10)
        )
    def forward(self,x):
        return self.module(x)
# 创建网络
net = Net()
net.to(device)
# 优化器
optimizer =optim.SGD(net.parameters(),lr=learning_rate)

# 损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn.to(device)

# 训练
total_train_step = 0
for i in range(epoch):
    for data in train_loader:
        imgs,targets = data
        imgs, targets = imgs.to(device),targets.to(device)
        outputs = net(imgs)
        loss =loss_fn(outputs,targets)
        total_train_step += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if total_train_step %100 ==0:
            print("第{}轮的loss:{}".format(total_train_step,loss))


# 测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
        print("整个测试集合上的loss为{}".format(total_test_loss))