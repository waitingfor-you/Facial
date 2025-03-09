import torch
import onnx

from model.VGGnet.joint import VGGnet

# 加载模型
model_path = r'C:\Users\yu\Desktop\facial\utils\models\level\model-0.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGnet()  # 替换为实际模型类别数
state_dict = torch.load(model_path, map_location=device)
# 因为是pt文件，所以需要直接使用load进行模型的加载
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 模拟输入
x = torch.randn(1, 1, 48, 48).to(device)

# 检查模型的实际输出
with torch.no_grad():
    output = model(x)
print(type(output))
if isinstance(output, (list, tuple)):
    print(len(output))

# 根据实际输出修改 output_list
if isinstance(output, torch.Tensor):
    output_list = ['output']
elif isinstance(output, (list, tuple)) and len(output) == 2:
    output_list = ['output1', 'output2']
else:
    raise ValueError("模型输出的数量与 output_list 不匹配")

# 导出 ONNX
torch.onnx.export(
    model,
    x,
    "./model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=output_list
)
