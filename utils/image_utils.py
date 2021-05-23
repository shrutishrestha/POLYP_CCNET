from networkx.algorithms.shortest_paths.unweighted import predecessor
import numpy as np
from PIL import Image
import os
import cv2
import torch


def get_val_merged_image(original_image, label, pred_mask, ccnet_upsampled):
    # label = label.cpu().numpy()
    label_copy = label.copy()
    label_copy[label == 1] = 255
    label_copy = np.asarray(label_copy, np.uint8)

    original_image = np.asarray(original_image, np.uint8)
    original_image = Image.fromarray(np.squeeze(original_image, axis=0))

    if len(label_copy.shape) == 3:
        label = Image.fromarray(np.squeeze(label_copy, axis=0))
    else:
        label = Image.fromarray(label_copy)

    pred_mask = torch.from_numpy(pred_mask)
    pred_mask = (pred_mask>0).float()
    pred_mask = pred_mask.cpu().detach().numpy()
    pred_mask_copy = pred_mask.copy()
    pred_mask_copy[pred_mask == 1] =255

    pred_mask = Image.fromarray(np.squeeze(pred_mask_copy, axis=0))


    ccnet_upsampled = torch.from_numpy(ccnet_upsampled)
    ccnet_upsampled = (ccnet_upsampled>0).float()
    ccnet_upsampled = ccnet_upsampled.cpu().detach().numpy()
    ccnet_upsampled_copy = ccnet_upsampled.copy()
    ccnet_upsampled_copy[ccnet_upsampled == 1] =255

    ccnet_upsampled = Image.fromarray(np.squeeze(ccnet_upsampled_copy, axis=0))


    total_width = label.size[0] + 10+ label.size[0] + 10+pred_mask.size[0] +10+ label.size[0]
    max_height = max(label.size[1], pred_mask.size[1])
    boader = 255 * np.ones((max_height,max_height))
    boader = Image.fromarray(boader) 
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(ccnet_upsampled, (x_offset,0))
    x_offset += ccnet_upsampled.size[0]

   #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(original_image, (x_offset,0))
    x_offset += original_image.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(label, (x_offset,0))
    x_offset += label.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(pred_mask,(x_offset,0))
    x_offset +=pred_mask.size[0]

    new_im_array = np.array(new_im)
    new_im_array = new_im_array.transpose((2,0,1))
    new_im_tensor = torch.from_numpy(new_im_array)

    return new_im_tensor


def get_train_merged_image(image, label, pred_mask):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = np.asarray(image.transpose((1,2,0)), np.uint8)
    image = Image.fromarray(image)

    label = label.cpu().numpy()
    label_copy = label.copy()
    label_copy[label == 1] = 255
    label_copy = np.asarray(label_copy, np.uint8)
    if len(label_copy.shape) == 3:
        label = Image.fromarray(np.squeeze(label_copy, axis=0))
    else:
        label = Image.fromarray(label_copy)
    pred_mask = torch.from_numpy(pred_mask)
    pred_mask = (pred_mask>0).float()
    pred_mask = pred_mask.cpu().detach().numpy()
    pred_mask_copy = pred_mask.copy()
    pred_mask_copy[pred_mask == 1] =255
    pred_mask = Image.fromarray(pred_mask_copy) 

    total_width = image.size[0]+10+label.size[0] + 10+pred_mask.size[0]
    max_height = max(image.size[1], label.size[1], pred_mask.size[1])
    boader = 255 * np.ones((max_height,max_height))
    boader = Image.fromarray(boader) 
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    new_im.paste(image, (x_offset,0))
    x_offset += image.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(label, (x_offset,0))
    x_offset += label.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(pred_mask,(x_offset,0))
    x_offset +=pred_mask.size[0]

    new_im_array = np.array(new_im)
    new_im_array = new_im_array.transpose((2,0,1))
    new_im_tensor = torch.from_numpy(new_im_array)

    return new_im_tensor










