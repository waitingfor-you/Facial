import os.path

path = '../fer2013/Training'
# 数据集文件夹所在路径，其中文件夹的名称为标签
txtPath = '../labels.txt'
# 生成的标签文件位置和名称

assert os.path.isdir(path), '该文件夹不存在！！！'

labels = os.listdir(path)
# print(labels)
with open(txtPath, 'w') as file:
    for i in range(len(labels)):
        file.write("{} {}\n".format(labels[i], i))
    file.write("none 7")