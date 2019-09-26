import matplotlib.pyplot as plt

import os
import shutil
import logging
import torch
import numpy as np
import torchvision.utils as vutils
from utils.record_db import start_expr
_join = os.path.join

def make_log(current_dir, result_dir, expeerment_name):
    EXPR_ID: int = start_expr(expeerment_name, '', '', '')
    print('EXPR_ID', EXPR_ID)
    log_dir = os.path.join(current_dir, result_dir, expeerment_name + '%s' % EXPR_ID)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # current_dir = os.path.abspath(os.getcwd())

    shutil.copytree(_join(current_dir, 'utils'), _join(log_dir, 'srcipt'))

    logging.basicConfig(level=logging.INFO,
                        filename=_join(log_dir, 'new.log'),
                        filemode='w',
                        format='%(asctime)s - : %(message)s')

    return log_dir, logging


def write_image(writer,dict_message, step, mode):
    for key in dict_message:
        data = torch.unsqueeze(dict_message[key][0, ...], 0)
        x = vutils.make_grid(data)
        writer.add_image('%s/%s' % (mode, key), x, step)

def finetune_load(Net, pkl_path):
    # load params
    pretrained_dict = torch.load(pkl_path)

    # 获取当前网络的dict
    net_state_dict = Net.state_dict()

    # 剔除不匹配的权值参数
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

    # 更新新模型参数字典
    net_state_dict.update(pretrained_dict_1)

    # 将包含预训练模型参数的字典"放"到新模型中
    Net.load_state_dict(net_state_dict)

def myprint(logging, message):
    logging.info(message)
    print(message)

def print_writer_scalar(writer, logging, dict_message, step, mode):
    if mode == 'train':
        message = 'Step: %s  ' % step
    else:
        message = '[******] Step: %s  ' % step
    for key in dict_message:
        message += key + ':  %s  ' % dict_message[key]
        writer.add_scalar('%s_result/%s' % (mode, key), dict_message[key], step)

    myprint(logging, message)

def print_writer_scalars(writer, dict_message_train, dict_message_test, step):
    for key in dict_message_test:
        writer.add_scalars('all_result/%s' % key,
                           {'train_%s'%key:dict_message_train[key],
                            'test_%s'%key:dict_message_test[key]}, step)

def show_confMat_utils(confusion_mat, classes_name, set_name, writer, out_dir, step):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(np.around(confusion_mat_N[i, j]*100)), va='center', ha='center', color='red', fontsize=10)
    # 保存
    save_path = os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png')
    plt.savefig(save_path)
    plt.close()
    image = np.asarray(plt.imread(save_path))
    x = vutils.make_grid(torch.from_numpy(image.transpose((2,0,1))[:3,...]))
    writer.add_image('Confusion_Matrix', x, step)

def show_confMat(predict, target, num_class, writer, log_dir, step):
    conf_mat = np.zeros([num_class, num_class])
    for i in range(num_class):
        for j in range(num_class):
            true_i = target.numpy()==i
            pre_i = predict.numpy()==j
            conf_mat[i, j] = np.sum(true_i[pre_i])
    show_confMat_utils(conf_mat, ['G', 'V'], 'vessel', writer, log_dir, step)
