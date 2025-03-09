import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch

a = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

b = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

# 图片路径
image_path = r"C:\Users\yu\Desktop\facial\data\train\anger\0.jpg"  # 替换为你的图片路径
image_path2 = r"C:\Users\yu\Desktop\facial\data\train\anger\0.jpg"  # 替换为你的图片路径

# 加载图片
image = Image.open(image_path).convert('RGB')  # 确保图片是 RGB 模式
image2 = Image.open(image_path2).convert('RGB')  # 确保图片是 RGB 模式

# 应用变换
transformed_image = a(image)
transformed_image2 = b(image2)

print("Transformed Tensor Data:", transformed_image)
print("Shape:", transformed_image.shape)  # 查看形状 (C, H, W)
print("Min Value:", torch.min(transformed_image))  # 最小值
print("Max Value:", torch.max(transformed_image))  # 最大值
print("~~~~~~~~~~~")
print("Transformed Tensor Data:", transformed_image2)
print("Shape:", transformed_image2.shape)  # 查看形状 (C, H, W)
print("Min Value:", torch.min(transformed_image2))  # 最小值
print("Max Value:", torch.max(transformed_image2))  # 最大值