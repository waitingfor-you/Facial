import os
import shutil

path = '../data'
# 具体训练集的位置
labeltxt = '../labels.txt'
dataset = '../fer2013/Training'
# 标签位置
with open(labeltxt, 'r') as file:
    lines = file.readlines()

cleaned_lines = []
for line in lines:
    line = line.replace(' ', '')
    line = ''.join([char for char in line if not char.isdigit()])
    cleaned_lines.append(line.strip())
print(cleaned_lines)

paths = [os.path.join(path, x) for x in ['train', 'val']]
# paths存储的是data下面的train集和val集
print(paths)
if not os.path.exists(paths[0]):
    os.mkdirs(paths[0])
if not os.path.exists(paths[1]):
    os.mkdirs(paths[1])
# 建立路径
for i in paths:
    for j in cleaned_lines:
        labelpath = os.path.join(i, j)
        if not os.path.exists(labelpath):
            os.mkdir(labelpath)

for j in cleaned_lines:
    if j == 'none': continue
    fromPath = os.path.join(dataset, j)
    file_list = os.listdir(fromPath)
    cnt = len(file_list)
    num = int(cnt * 0.7)     # 训练集占比
    toPath = os.path.join(path, 'train', j)
    for index in range(0, num):
        source = os.path.join(fromPath, file_list[index])
        shutil.copy(source, toPath)
    toPath = os.path.join(path, 'val', j)
    for index in range(num, cnt):
        source = os.path.join(fromPath, file_list[index])
        shutil.copy(source, toPath)



print('ok')