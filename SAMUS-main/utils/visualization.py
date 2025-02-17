import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
import seaborn as sns


def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) 
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_segmentation_sets(seg, image_filename, opt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    #img = img_ori0
          
    fulldir = opt.result_path + "/" + opt.modelname + "/"
    #fulldir = opt.result_path + "/" + "GT" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    #img = img_ori0
    
    pt = np.array(pt.cpu())
    N = pt.shape[0]
    # for i in range(N):
    #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 6, (0,0,0), -1)
    #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 5, (0,0,255), -1)
    #     cv2.line(img, (int(pt[i, 0]-3), int(pt[i, 1])), (int(pt[i, 0])+3, int(pt[i, 1])), (0, 0, 0), 1)
    #     cv2.line(img, (int(pt[i, 0]), int(pt[i, 1])-3), (int(pt[i, 0]), int(pt[i, 1])+3), (0, 0, 0), 1)
          
    fulldir = opt.result_path + "/PT10-" + opt.modelname + "/"
    #fulldir = opt.result_path + "/PT3-" + "img" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_compare(image_filename,pred,gt,opt,epoch):
    # 原始图片
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    gt_pic = gt[0,:,:]
    pred_pic = pred[0,:,:]

    # np.where 返回的是满足条件的像素索引 (row, col)
    crop_pic = np.zeros_like(img_ori)
    mask = (pred_pic==1.0)
    # #
    # # # 2. 将掩码图中为 255 的位置的原图像素值复制到 modified_img 中
    # for row in range(crop_pic.shape[0]):
    #     for col in range(crop_pic.shape[1]):
    #         if pred_pic[row,col]==1.0:
    #             crop_pic[:, row,col] = img_ori[:, row,col]
    crop_pic[mask,:] = img_ori[mask,:]
    print(image_filename)
    # 创建一个图形窗口
    plt.figure(figsize=(12, 3))
    # 显示第一个掩码图
    plt.subplot(1, 4, 1)
    plt.imshow(img_ori)
    plt.title('ori_image')
    plt.axis('off')

    # 显示第二个掩码图
    plt.subplot(1, 4, 2)
    plt.imshow(gt_pic,cmap='gray')
    plt.title('gt_mask')
    plt.axis('off')

    # 显示第三个掩码图
    plt.subplot(1, 4, 3)
    plt.imshow(pred_pic,cmap='gray')
    plt.title('pred_mask')
    plt.axis('off')

    # 显示第四个掩码图
    plt.subplot(1, 4, 4)
    plt.imshow(crop_pic)
    plt.title('crop_pic')
    plt.axis('off')
    # 显示图像
    # plt.tight_layout()
    # plt.show()
    # 指定保存目录和文件名
    if 'KTD' in opt.data_path:
        output_dir = opt.result_path + "/Merge-" + opt.modelname +"/KTD"+ "/" + str(epoch) + "/"  # 指定目录
    else:
        output_dir = opt.result_path + "/Merge-" + opt.modelname + "/"+str(epoch)+"/"  # 指定目录

    output_file = image_filename  # 指定文件名
    # fulldir = opt.result_path + "/PT3-" + "img" + "/"
    # 创建目录（如果目录不存在）
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    # 保存图像到指定目录
    save_path = os.path.join(output_dir, output_file)
    plt.savefig(save_path)
    plt.close()

def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)