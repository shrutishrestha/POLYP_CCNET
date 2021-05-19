from networkx.algorithms.shortest_paths.unweighted import predecessor
import numpy as np
from PIL import Image
import os
import cv2
import torch


def merge_image( image, label, pred_mask):
        #change to numpy
        pred_mask = np.squeeze(pred_mask.cpu().detach().numpy(), axis=0)

        #change 1 to 255
        pred_tensor_mask_copy = pred_mask.copy()
        pred_tensor_mask_copy[pred_mask == 1] =255
        pred_mask = Image.fromarray(pred_tensor_mask_copy) 

        #specify total height and width after merging
        total_width = image.size[0]+10+label.size[0] + 10+pred_mask.size[0]
        max_height = max(image.size[1], label.size[1], pred_mask.size[1])

        #initialize for boader
        boader = 255*np.ones((max_height,max_height))
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

        return new_im
        
def frame_and_save_image(image_name, image, label, pred_mask, output_path, new_im):
    merged_image = merge_image(output_path, image, label, pred_mask)
    output_path = os.path.join(output_path,image_name+".jpg") #current_image_name
    merged_image.save(output_path) 


def scale_image(output, size, padded_side, value_to_scale, input_size):
    h, w = map(int, input_size.split(','))
    input_size = (h, w)
    ori_image_height = size[0][0]
    ori_image_width = size[0][1]
    input_height = input_size[0]
    input_width = input_size[1]

    output = torch.squeeze(output)
    output = output.cpu().numpy()

    if "h" in padded_side.keys():

        image_height = input_height - padded_side.get("h")
        output = np.asarray(output[0: image_height, 0: input_width], np.float32) 

    elif "w" in padded_side.keys():
        image_width = input_width - padded_side.get("w")
        output = np.asarray(output[0: input_height, 0: image_width], np.float32) 

    inverse = 1/value_to_scale
    inverse = round(inverse.item(), 6)

    output = cv2.resize(output, None, fx=inverse, fy=inverse, interpolation = cv2.INTER_LINEAR)

    if output.shape[0] != size[0][0] or output.shape[1] != size[0][1]:
        output = cv2.resize(output, (ori_image_width, ori_image_height ), interpolation = cv2.INTER_LINEAR)
    output = np.expand_dims(output, axis=0)
    output = torch.from_numpy(output)
    return output


def save_image(output_path, name, scaled_pred_mask,val_label, metric,data_path):
    current_image_name = name[0]
    image_ori = Image.open(data_path + "/images/" + current_image_name+".jpg")
    label_ori = Image.open(data_path + "/masks/" + current_image_name+".jpg")
    pred_tensor_mask = scaled_pred_mask.cpu().numpy()
    pred_tensor_mask_copy = pred_tensor_mask.copy()
    pred_tensor_mask_copy[pred_tensor_mask == 1] =255
    pred_mask = Image.fromarray(np.squeeze(pred_tensor_mask_copy,axis=0)) 


    val_label = val_label.cpu().numpy()
    val_label_copy = val_label.copy()
    val_label_copy[val_label == 1] = 255
    label_ori = Image.fromarray(np.squeeze(val_label_copy, axis=0))


    total_width = image_ori.size[0]+10+label_ori.size[0] + 10+pred_mask.size[0]
    max_height = max(image_ori.size[1], label_ori.size[1], pred_mask.size[1])
    boader = 255*np.ones((max_height,max_height))
    boader = Image.fromarray(boader) 
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    new_im.paste(image_ori, (x_offset,0))
    x_offset += image_ori.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(label_ori, (x_offset,0))
    x_offset += label_ori.size[0]

    #border
    new_im.paste(boader, (x_offset,0))
    x_offset += 10
    new_im.paste(pred_mask,(x_offset,0))
    x_offset +=pred_mask.size[0]

    output_path = os.path.join(output_path,str(round(metric,5))+"_"+current_image_name+".jpg") #current_image_name
    new_im.save(output_path) 


def get_val_merged_image(label, pred_mask):

    # label = label.cpu().numpy()
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

    pred_mask = Image.fromarray(np.squeeze(pred_mask_copy, axis=0))

    total_width = label.size[0] + 10+pred_mask.size[0]
    max_height = max(label.size[1], pred_mask.size[1])
    boader = 255 * np.ones((max_height,max_height))
    boader = Image.fromarray(boader) 
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0

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










