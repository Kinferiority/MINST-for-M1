import torch
import torchvision.transforms
from PIL import Image
from torch import nn
import os


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


root_dir = "test_number"
number_name = "test11.png"
img_path = os.path.join(root_dir,number_name)
img = Image.open(img_path)
# img.show()
# 将三通道图片转换成单通道图片
img_1 = img.convert('1')
trans_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.reshape(x, (1, 1, 32, 32)))  # 在这里添加形状变换
])

img_1 = trans_pose(img_1)
print(img_1.shape)


mynet = torch.load('MNIST_24_acc_0.9843999743461609.pth', map_location=torch.device('cpu'))

mynet.eval()
with torch.no_grad():
    output = mynet(img_1)
    number = output.argmax(axis=1).item()
    print(f'识别的数字是{number}')

# 该模型对于具有三通道的黑白图片的拟合效果较好，而对三通道的彩色图片拟合效果较差