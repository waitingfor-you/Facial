import argparse
import os.path
import time

import torchvision
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model.VGGnet.joint import VGGnet



parser = argparse.ArgumentParser('train parser')
parser.add_argument('--level', help='data augment level', type=int, default=0)
parser.add_argument('--pretrained', help='pretrained path', type=str, default='./models/level/model-0.pt') # 这里放入预训练模型路径
parser.add_argument('--modelpath', help='saved model path', type=str, default='models/level')
parser.add_argument('--logpath', help='saved log path', type=str, default='logs/level')
parser.add_argument('--flag', help='saved log path', type=str, default='logs/level')
args = parser.parse_args()

writer = SummaryWriter(args.logpath)

image_size = 48
crop_size = 48
data_transforms = {}
data_transforms['val'] = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(crop_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

data_transforms_train_list = []

# 按照等级来选择数据加强
if args.level < 1:
    data_transforms_train_list.append(transforms.Resize(crop_size))
if args.level >= 1:
    data_transforms_train_list.append(transforms.RandomResizedCrop(crop_size))
    data_transforms_train_list.append(transforms.RandomHorizontalFlip())
if args.level >= 2:
    data_transforms_train_list.append(transforms.RandomRotation(15))
if args.level >= 3:
    data_transforms_train_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

data_transforms_train_list.append(transforms.Grayscale(num_output_channels=1))
data_transforms_train_list.append(transforms.ToTensor())
data_transforms_train_list.append(torchvision.transforms.Normalize([0.5], [0.5]))
data_transforms['train'] = transforms.Compose(data_transforms_train_list)

def train_model(model, criterion, optimizer, scheduler, num_epoch=25):
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch-1))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_accs = 0.0
            number_batch = 0.0
            start_time = time.time()
            for data in dataloaders[phase]:

                inputs, labels = data[0], data[1]
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                # 这里的话，outputs是对应的概率，后续的labels的内容就是对应的图片的标签，然后使用这个loss的计算不需要进行softmax
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_accs += torch.sum(preds == labels).item()
                running_loss += loss.data.item()
                number_batch += 1
            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} loss: {:.4f} Acc:{:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))
            end_time = time.time()

            # 计算并输出运行时间
            elapsed_time = end_time - start_time
            print(f"程序运行时间：{elapsed_time:.4f} 秒")
        if(epoch % 50 == 0):
            torch.save(model.state_dict(), os.path.join(args.modelpath, 'model-{}.pt'.format(epoch)))
    writer.close()
    return model

def load_pretrained_model(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        print("文件路径不存在！！！")
        return
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint, strict=False)  # strict=False 允许部分加载
    print("Pretrained weights loaded successfully.")

if __name__ == '__main__':
    nclass = 8
    # 分类数量
    model = VGGnet()
    data_dir = '../data'
    # 数据目录

    if not os.path.exists('models'):
        os.mkdir('models')

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    # 建议在加载预训练文件前丢进cuda，加载速度更快
    print(model)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x]) for x in ['train',  'val']}
    # ImageFolder 会根据子文件夹的名称自动为图像分配标签。比如，class_1 里的图片会被标记为标签 0，class_2 里的图片会被标记为标签 1，依此类推。
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4) for x in ['train',  'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train',  'val']}

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    pretrained_path = args.pretrained  # 预训练文件路径
    load_pretrained_model(model, pretrained_path)

    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epoch=300)
    torch.save(model.state_dict(), os.path.join(args.modelpath, 'modelFinal.pt'))


