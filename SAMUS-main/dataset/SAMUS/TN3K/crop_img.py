import os
import cv2
import numpy as np
from utils.data_us import JointTransform2D, ImageToImage2D
if __name__ == '__main__':
    img_path = "./img"
    label_path  = "./label"
    pics = os.listdir(img_path)
    crop_img_path = "./crop_img"
    if not os.path.isdir(crop_img_path):
        os.makedirs(crop_img_path)
    for pic in pics:
        image_name = pic
        image = cv2.imread(os.path.join(img_path, image_name))
        img_ori = cv2.resize(image, dsize=(256, 256))

        mask = cv2.imread(os.path.join(label_path, image_name), 0)
        # mask = cv2.resize(mask, dsize=(256, 256))
        joint_transform = JointTransform2D(img_size=256, low_img_size=128,
                                    ori_size=256, crop=None, p_flip=0.0, p_rota=0.5, p_scale=0.5,
                                    p_gaussn=0.0,
                                    p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None,
                                    long_mask=True)  # image reprocessing
        _, mask, _ = joint_transform(image, mask)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1

        crop_pic = np.zeros_like(img_ori)
        mask = (mask == 1)
        crop_pic[mask,:] = img_ori[mask,:]
        cv2.imwrite(os.path.join(crop_img_path, image_name),crop_pic)
        print(pic)
