import torch
import torchvision.transforms
from matplotlib.pyplot import plot
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt

train_loss = list()
test_loss = list()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 下载数据
train_data = datasets.MNIST(root="data", train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
test_data = datasets.MNIST(root="data", train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
# 加载数据
batch_size = 64
train_data_set = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_set = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练集长度：{train_data_size}")
print(f"训练集长度：{test_data_size}")


# 创建网络模型
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        return self.module(x)


# 初始化网络
mynet = MyNet()
mynet = mynet.to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
lr = 0.01
optimzier = optim.SGD(mynet.parameters(),lr=lr)
epoch =25
for i in range(epoch):
    # 训练
    mynet.train()
    train_step = 0
    for imgs,targets in train_data_set:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = mynet(imgs)
        loss = loss_fn(outputs,targets)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        train_step += 1
        if train_step % 100 == 0:
            # print(imgs.shape)
            # print(outputs.shape)
            print(f"第{train_step}次训练的loss为:{loss}")
            train_loss.append(loss)


    # 测试
    accuarcy = 0
    total_accuracy = 0
    mynet.eval()
    with torch.no_grad():
        for j,(imgs,targets) in enumerate(test_data_set):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mynet(imgs)
            loss = loss_fn(outputs,targets)
            accuracy = (outputs.argmax(axis=1) == targets).sum()
            total_accuracy += accuarcy
            if j % 100 == 0:
                test_loss.append(loss)
    print(f"第{i+1}轮的准确率为{total_accuracy/test_data_size}")
    # 保存模型
    torch.save(mynet, f'MNIST_{i}_acc_{total_accuracy / test_data_size}.pth')

train_loss = [loss.item() for loss in train_loss]
test_loss = [loss.item() for loss in test_loss]

# 绘制训练和测试损失曲线
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
