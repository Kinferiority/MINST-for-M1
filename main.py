import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms


# 定义超参数
Train_BATH_SIZE = 64
TEST_BATH_SIZE = 1000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
epoch = 20
learing_rate = 0.01
total_train_step = 0
total_test_step = 0

# 构建pipeline ，图像处理
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 正则化，降低模型复杂度
])

writer = SummaryWriter("./logs_train")
# 下载和加载数据
from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=Train_BATH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=TEST_BATH_SIZE, shuffle=False)
# 数据集长度
train_set_size = len(train_set)
test_set_size = len(test_set)


# 构建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self, x):
        return self.model(x)

# 创建网络模型
net = Net()
net.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
optimizer =optim.SGD(net.parameters(), lr=learing_rate)

for i in range(epoch):
    print("-----第{}轮训练开始------".format(i+1))
    # 训练开始
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step+1
        if total_train_step%100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step, loss))
            writer.add_scalar("tran_loss", loss, total_train_step)
    # 测试
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accurary = (outputs.argmax(1) == targets).sum()
            total_accuracy += accurary
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_set_size))
    writer.add_scalar("test_accuracy", total_accuracy/test_set_size, total_test_step)
    writer.add_scalar("test_loss", total_test_loss,total_test_step)
    total_test_step = total_test_step+1
    torch.save(net,"net_{}.pth".format(i))
    print("模型已保存")
writer.close()




