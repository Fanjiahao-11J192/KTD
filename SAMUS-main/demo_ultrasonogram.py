from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model,get_classifier
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from utils.data_us import *
from thop import profile
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import cv2
import matplotlib.pyplot as plt
def visual_compare(pred,image_path,args):
    # 原始图片
    image_filename = image_path.split("/")[-1]
    img_ori = cv2.imread(image_path)
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    pred_pic = pred[:,:]

    # np.where 返回的是满足条件的像素索引 (row, col)
    crop_pic = np.zeros_like(img_ori)
    mask = (pred_pic==255)
    # #
    # # # 2. 将掩码图中为 255 的位置的原图像素值复制到 modified_img 中
    # for row in range(crop_pic.shape[0]):
    #     for col in range(crop_pic.shape[1]):
    #         if pred_pic[row,col]==1.0:
    #             crop_pic[:, row,col] = img_ori[:, row,col]
    crop_pic[mask,:] = img_ori[mask,:]

    # 创建一个图形窗口
    plt.figure(figsize=(9, 3))
    # 显示第一个掩码图
    plt.subplot(1, 3, 1)
    plt.imshow(img_ori)
    plt.title('ori_image')
    plt.axis('off')

    # 显示第三个掩码图
    plt.subplot(1, 3, 2)
    plt.imshow(pred_pic,cmap='gray')
    plt.title('pred_mask')
    plt.axis('off')

    # 显示第四个掩码图
    plt.subplot(1, 3, 3)
    plt.imshow(crop_pic)
    plt.title('crop_pic')
    plt.axis('off')
    # 显示图像
    # plt.tight_layout()
    # plt.show()
    # 指定保存目录和文件名
    output_dir = args.result_path + "/ultrasonogram/"  # 指定目录
    output_file = image_filename  # 指定文件名
    # fulldir = opt.result_path + "/PT3-" + "img" + "/"
    # 创建目录（如果目录不存在）
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 保存图像到指定目录
    save_path = os.path.join(output_dir, output_file.split('.')[0]+'_compare.jpg')
    plt.savefig(save_path)
    plt.close()

    save_path2 = os.path.join(output_dir+'', output_file.split('.')[0]+'_crop.jpg')
    cv2.imwrite(save_path2,crop_pic)
    print('保存' + output_file + '的分割图像到：', save_path2)
    return save_path2
def origin_data_preprocess(image_path,args):
    image = cv2.imread(image_path,0)

    image = correct_dims(image)
    joint_transform = JointTransform2D(img_size=256, low_img_size=128,
                                       ori_size=256, crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    mask = np.zeros_like(image) # 初始化一个mask占位
    image, _, _ = joint_transform(image, mask)
    image = image.unsqueeze(0).to(dtype = torch.float32, device=args.device)

    return image

def crop_data_preprocess(crop_image_path,args):
    image = cv2.imread(crop_image_path)
    # cv2.imshow('image',image)
    image = correct_dims(image)
    joint_transform = JointTransform2D(img_size=256, low_img_size=128,
                                       ori_size=256, crop=None, p_flip=0.0, color_jitter_params=None,long_mask=True)  # image reprocessing
    mask = np.zeros_like(image)  # 初始化一个mask占位
    image, _, _ = joint_transform(image, mask)
    image = image.unsqueeze(0).to(dtype=torch.float32, device=args.device)

    return image

def pred_mask_postprocess(pred_mask,image_path,args):
    predict = torch.sigmoid(pred_mask['masks'])
    predict = predict.detach().cpu().numpy()  # (b, c, h, w)
    seg = predict[0, 0, :, :] > 0.5  # (b, h, w)
    h, w = seg.shape

    pred = np.zeros((h, w))
    pred[seg[:, :] == 1] = 255

    # # 保存图片
    crop_picture_path = visual_compare(pred,image_path,args)
    return crop_picture_path

def pred_class_postprocess(pred_class):
    _, predicted = torch.max(pred_class, dim=1)  # 选择最大概率对应的类标签
    pred_class = predicted.cpu().numpy()[0]
    return pred_class

def demo(data_path,model_path):

    #  =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str,
                        help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('--encoder_input_size',default=256,type=int,help='the input size of SAMUS')
    parser.add_argument("--classifier_name", default='Resnet18', type=str,
                        help='type of classifier, e.g., Restnet18, Vit...')
    parser.add_argument("--classifier_size", default=256, type=int,
                        help='the input size of classifier')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--SAMUS_ckpt',default='SAMUS_01101206_340_0.8371178529308249.pth',type=str,help='Segment_model_save_path')
    parser.add_argument('--classifier_ckpt',default='Resnet18_01032018_59_0.7805483249970666.pth',type=str,help='Classifier_model_save_path')
    parser.add_argument('--classifier_classes',default=2,type=int,help='the number of classes')
    parser.add_argument('--batch_size',type= int,default= 1 ,help='the batch number of data')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--result_path',type=str,default='./result/demo/',help='the path of saved pred mask pictures')
    args = parser.parse_args()


    # 加载模型
    device = torch.device(args.device)

    # 加载分割模型
    segment_modal = get_model(args.modelname, args=args)
    segment_modal.to(device)
    segment_modal.train()
    checkpoint = torch.load(model_path+'/'+args.SAMUS_ckpt)
    # ------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    segment_modal.load_state_dict(new_state_dict)
    segment_modal.eval()
    print("加载分割模型完成...")
    # 加载分类模型
    classifier = get_classifier(opt=args)
    classifier.to(device)
    classifier.train()
    checkpoint = torch.load(model_path+'/'+args.classifier_ckpt)
    # ------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    classifier.load_state_dict(new_state_dict)
    classifier.eval()
    print("加载分类模型完成...")

    # 加载数据

    image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print('读取数据完成')
    print(image_files)

    for image_path in image_files:
        image_name = image_path.split("/")[-1]
        print('预处理图片：',image_name)
        image = origin_data_preprocess(image_path,args)

        print('分割图片:', image_name)
        pred_mask = segment_modal(image)

        # 保存掩码图片
        crop_image_path = pred_mask_postprocess(pred_mask,image_path,args)
        crop_image_name  = crop_image_path.split("/")[-1]
        # 加载分割图片，进行分类
        print('预处理分割图片:',crop_image_name)
        crop_image = crop_data_preprocess(crop_image_path,args)

        #执行分类
        pred_class= classifier(crop_image)
        pred_rs =pred_class_postprocess(pred_class)
        #
        print(image_name+'的分类结果为:',pred_rs)

if __name__ == '__main__':
    data_path = "dataset/SAMUS/TN3K/demo_img"
    model_path = "checkpoints/TN3K"
    demo(data_path,model_path)
