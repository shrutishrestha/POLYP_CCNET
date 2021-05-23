import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import os
import torch
from engine import Engine
import matplotlib.pyplot as plt
import imageio
import numpy as np


class KvasirSegDataSet(data.Dataset):
    def __init__(self, data_dir, max_iters=None, crop_size=(769,769), mean=(104.00698793,116.66876762,122.67891434), scale=True,
                 mirror=True, ignore_label=255, test=False):
        self.data_dir = data_dir
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = self.get_img_ids(data_dir)
        self.files = []
        self.test = test
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            
        for item in self.img_ids:
            image_path, label_path = item

            name = osp.splitext(osp.basename(label_path))[0]
            self.files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
                              

    def get_img_ids(self, data_dir):
        image_names = os.listdir(os.path.join(data_dir, "images"))
        total_list_path = []
        for img_name in image_names:
            full_img_path = os.path.join(data_dir, "images", img_name)
            full_mask_path = os.path.join(data_dir, "masks", img_name)
            total_list_path.append((full_img_path, full_mask_path))
        return total_list_path


    def __len__(self):
        return len(self.files)


    def generate_scale_label(self, image, label):
        f_scale = 0.7 + random.randint(0,14) / 10.0
        image = cv2.resize(image, None, fx = f_scale, fy = f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx = f_scale, fy = f_scale, interpolation= cv2.INTER_NEAREST)
        return image, label


    def id2trainId(self, label):
        label_copy = label.copy()
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v
        label_copy[label!=0] = 1 #edited 
        return label_copy

        
    def __getitem__(self, index):

        datafiles = self.files[index]
        name = datafiles["name"]
        ori_image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            label = np.asarray(label, np.float32)
            ori_image = np.asarray(ori_image, np.float32)
            image = ori_image - self.mean
            image = image.transpose((2, 0, 1))

            return ori_image, image.copy(), label.copy(), np.array(size), name


