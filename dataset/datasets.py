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
                 mirror=True, ignore_label=1, cutmix = True, test=False):
        self.data_dir = data_dir
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = self.get_img_ids(data_dir)
        self.files = []
        self.cutmix = cutmix
        self.test = test

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
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)
        
        size = image.shape
        
        if not self.test:
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
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
            image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
            #image = image[:, :, ::-1]  # change to BGR
            image = image.transpose((2, 0, 1))
            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                image = image[:, :, ::flip]
                label = label[:, ::flip]

        else:
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image -= self.mean
            image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


