import torch
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
image_path = "img.png"
image = Image.open(image_path)
print(image)
image = image.convert("L")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor()
])
image = transform(image)
print(image.shape)
# 将张量转换为 NumPy 数组
image_array = image.numpy()

# 显示图像
plt.imshow(image_array[0], cmap="gray")
plt.axis("off")  # 去除坐标轴
plt.show()


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

model = torch.load("net_19.pth",map_location=torch.device('cpu'))
print(model)

image = torch.unsqueeze(image, 0)

model.eval()

with torch.no_grad():
    outputs = model(image)
    print(outputs)
    print(outputs.argmax(1))
