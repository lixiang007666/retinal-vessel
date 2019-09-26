import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import configparser
import os
_join = os.path.join
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils.data_utils import *
from utils.SF_utils import *
from utils.layer_utils import *
from utils.mydataset import *
from utils.metrics_utils import *
from model.UNet import Unet
from sklearn.metrics import *
# from model.S_UNet import Mi_UNet as Unet
# ----------------------------------------------------------------------------------------------------
# ------------------------------------ step 0 : 获取参数-----------------------------------------------
config = configparser.ConfigParser()
config.read('configuration.txt')
data_type = config.get('data attributes', 'dataset')
fold = config.get('data attributes', 'fold')

result_dir = config.get('result paths', 'result_dir')
expeerment_name = config.get('result paths', 'expeerment_name')

train_bs = int(config.get('training settings', 'train_bs'))
test_bs = int(config.get('training settings', 'test_bs'))
max_epoch = int(config.get('training settings', 'N_epochs'))
num_class = int(config.get('training settings', 'num_class'))
current_dir = os.path.abspath(os.getcwd())
finetune = config.get('training settings', 'finetune')
pkl_path = config.get('training settings', 'pkl_path')
# ----------------------------------------------------------------------------------------------------
# ------------------------------------ step 1 : 加载数据-----------------------------------------------
if data_type == 'CHASE':
    npz_directory = '%s/%s/processed/result_gray_512_16.h5' % (data_type, fold)
else:
    npz_directory = '%s/processed/result_gray_512_16.h5' % data_type

data_path = _join(current_dir, npz_directory)
if not os.path.isfile(data_path):
    source_data_dir = data_path.replace('processed/result_gray_512_16.h5', '')
    prepare_data(source_data_dir, data_path)

# data = get_dataset(data_path)
data_path = data_path.replace('.h5', '.npz')
data = np.load(data_path)

train_data = MyDataset((data['X_train'], data['y_train'], data['y_train_'], data['z_train']), transforms.ToTensor())
test_data = MyDataset((data['X_test'], data['y_test'], data['y_test_'], data['z_test']), transforms.ToTensor())
# 构建DataLoder#https://www.cnblogs.com/JeasonIsCoding/p/10168753.html
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs)

# ----------------------------------------------------------------------------------------------------
# ------------------------------------ step 2 : 定义网络-----------------------------------------------
Net = Unet(in_channels=1, num_class=num_class)
# initialize_weights(Net)    # 初始化权值
Net.cuda()#使用GPU
print(Net)
# ============================= ##        finetune 权值初始化
if finetune=='1':
    print('Load Model from : %s' % pkl_path)
    finetune_load(Net, pkl_path)
# ----------------------------------------------------------------------------------------------------
# ------------------------------------ step 3 : 定义损失函数和优化器 ------------------------------------
criterion = nn.CrossEntropyLoss()                               # 选择损失函数
optimizer0 = optim.Adam(Net.parameters(), lr=0.0001, weight_decay=0.00002)    # 选择优化器

scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=50, gamma=0.1)     # 设置学习率下降策略
# ----------------------------------------------------------------------------------------------------
# ------------------------------------ step 4 : 训练 --------------------------------------------------
# ================================ ##        新建writer
log_dir, logging = make_log(current_dir, result_dir, expeerment_name)
writer = SummaryWriter(log_dir=log_dir)
DiceLoss = DiceLoss()
for epoch in range(max_epoch):
    scheduler0.step()  # 更新学习率

    current_result = 0
    loss_sigma = []  # 记录一个epoch的loss之和
    accuracy = []
    Dice = []
    for step, data in enumerate(train_loader):
        # 获取图片和标签
        image, label_av, label_v, mask = data
        inputs, labels_v = Variable(image.cuda()), Variable(label_v.cuda().type(torch.long))
        #https://blog.csdn.net/VictoriaW/article/details/72673110
        # ================================ ##        优化
        optimizer0.zero_grad()
        outputs = Net(inputs)
        loss = criterion(outputs, labels_v)
        loss.backward()
        optimizer0.step()

        # 统计预测信息
        outputs = outputs.cpu()
        _, predict = torch.max(outputs.data, 1)
        acc, dice = metrics(predict, label_v)
        accuracy.append(acc)

        # label = label_v.cuda().type(torch.long)
        # dice = DiceLoss.forward(predict, label)
        Dice.append(dice)
        loss_sigma.append(loss.item())
    # ================================ ##        相关结果指标显示及记录
    dict_message_train = {'loss':np.mean(loss_sigma), 'accuracy':np.mean(accuracy),'dice':np.mean(Dice)}
    print_writer_scalar(writer,logging, dict_message_train, epoch, 'train')
    # ================================ ##        相关图片的记录
    dict_message = {'image':image, 'mask':mask*255, 'label_v':label_v*255, 'predict':predict}
    write_image(writer, dict_message, epoch, 'train')
    # ================================ ##        更新writer缓存区
    writer.flush()


    # ================================ ##        模型测试
    accuracy = []
    Roc = []
    Dice = []
    Auc = []
    for step, data in enumerate(test_loader):
        # 获取图片和标签
        image, label_av, label_v, mask = data
        outputs = Net(image.cuda())
        #统计预测信息
        _, predict = torch.max(outputs.cpu().data, 1)
        predicted = predict.numpy().squeeze()
        masked = mask.numpy().squeeze()
        predicted = predicted[masked==1]
        labeled = label_v.numpy().squeeze()
        labeled = labeled[masked == 1]
        dice1 = (np.sum(predicted*labeled*2)/(np.sum(predicted)+np.sum(labeled)))
        acc = (np.sum(predicted*labeled)/(np.sum(predicted)))
        # acc, dice = metrics(predict, label_v)
        p = outputs.cpu().data[0,1,:,:]
        p = p.numpy()[masked == 1]
        auc = roc_auc_score(y_true=labeled,y_score=p)
        # dice = hard_dice_coe(label_v.cuda(), outputs, num_class)

        # label = label_v.cuda().type(torch.long)
        # dice = DiceLoss.forward(predict, label)
        accuracy.append(acc)
        Dice.append(dice1)
        Auc.append(auc)
    # ================================ ##        相关结果指标显示及记录
    mean_dice = np.mean(Dice)
    dict_message_test = {'loss':np.mean(Auc),'accuracy':np.mean(accuracy),'dice':mean_dice}
    print_writer_scalar(writer, logging, dict_message_test, epoch, 'test')
    print_writer_scalars(writer, dict_message_train, dict_message_test, epoch)


    dict_message = {'image': image, 'mask': mask * 255, 'label_v': label_v * 255, 'predict': predict}
    write_image(writer, dict_message, epoch, 'test')
    show_confMat(predict.cpu(), label_v, num_class, writer, log_dir, epoch)
    # ================================ ##        模型保存
    if current_result < mean_dice:
        # net_save_path = os.path.join(log_dir, 'net_params_%s.pkl' % epoch)
        net_save_path = os.path.join(log_dir, 'net_params.pkl')
        torch.save(Net.state_dict(), net_save_path)

    writer.flush()
# ================================ ##        关闭writer
writer.close()