import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import tqdm
import networks
from PIL import Image
from torch.nn import functional as F
from engine import Engine
from metric import calculate_metrics
import numpy as np
from utils.image_utils import get_train_merged_image
from metric import get_confusion_matrix
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
    

def train_method(epoch, args, criterion, engine, global_iteration, model, optimizer, train_loader, train_sampler, summary_writer):
    model.train()
    if engine.distributed:
        train_sampler.set_epoch(epoch)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    train_pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                      bar_format=bar_format)
    train_loader_iterator = iter(train_loader)
    train_loss_sum = 0
    confusion_matrix_xdsn = np.zeros((args.num_classes,args.num_classes))
    confusion_matrix_ccnet = np.zeros((args.num_classes,args.num_classes))

    for idx in train_pbar:
        global_iteration += 1

        train_images, train_labels, _, name = train_loader_iterator.next()

        train_images = train_images.to(device)
        train_labels = train_labels.long().to(device)

        optimizer.zero_grad()
        num_steps = len(train_loader)* args.max_epochs
        lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration - 1, num_steps, args.power)

        train_output = model(x=train_images, labels=train_labels)  
        loss = criterion(preds=train_output, target=train_labels)

        reduce_loss = loss.data
        train_loss = reduce_loss.item()
        train_loss_sum += train_loss
        loss.backward()
        optimizer.step()

        # calculating metrics
        h, w = train_labels.size(1), train_labels.size(2)

        # calculating dice for ccnet_out
        ccnet_out = train_output[0]
        ccnet_out = F.interpolate(input=ccnet_out, size=(h, w), mode='bilinear', align_corners=True)
        ccnet_out = np.asarray(np.argmax(ccnet_out.cpu().detach().numpy(), axis=1), dtype=np.uint8) #(1, 1024, 2048)
        confusion_matrix_ccnet += get_confusion_matrix(gt_label = train_labels, pred_label =ccnet_out, class_num = args.num_classes, ignore_label=args.ignore_label)

        # calculating dice for xdsn
        xdsn_out = train_output[1]
        xdsn_out = F.interpolate(input=xdsn_out, size=(h, w), mode='bilinear', align_corners=True)
        xdsn_out = np.asarray(np.argmax(xdsn_out.cpu().detach().numpy(), axis=1), dtype=np.uint8) #(1, 1024, 2048)
        confusion_matrix_xdsn += get_confusion_matrix(gt_label = train_labels, pred_label = xdsn_out, class_num = args.num_classes, ignore_label=args.ignore_label)

        print_str = ' Iter{}/{}:'.format(idx + 1, len(
            train_loader)) + ' lr=%.2e' % lr + ' loss=%.2f' % reduce_loss.item()
        

        merged_image = get_train_merged_image(train_images[0], train_labels[0], ccnet_out[0])

        if idx==100 or idx == 500:
            summary_writer.add_image(tag="train_"+name[0], img_tensor = merged_image, global_step=epoch)
        
        train_pbar.set_description(print_str, refresh=False)
        torch.cuda.empty_cache()

        
    tn_train_ccnet, fp_train_ccnet, fn_train_ccnet, tp_train_ccnet, meanIU_train_ccnet, dice_train_ccnet, prec_ccnet, recall_ccnet = calculate_metrics(confusion_matrix_ccnet)
    tn_train_dsn, fp_train_dsn, fn_train_dsn, tp_train_dsn, meanIU_train_dsn, dice_train_dsn, prec_dsn, recall_dsn = calculate_metrics(confusion_matrix_xdsn)

    train_loss = round(train_loss_sum/len(train_loader),6)

    ccnet_train_metric = {
        "tn" : tn_train_ccnet,
        "fp" : fp_train_ccnet, 
        "fn" : fn_train_ccnet,
        "tp" : tp_train_ccnet,
        "meanIU" : meanIU_train_ccnet,
        "dice" : dice_train_ccnet,
        "precision" : prec_ccnet, 
        "recall" : recall_ccnet
    }
    
    dsn_train_metric = {
        "tn" : tn_train_dsn,
        "fp" : fp_train_dsn,
        "fn" : fn_train_dsn,
        "tp" : tp_train_dsn,
        "meanIU" : meanIU_train_dsn, 
        "dice" : dice_train_dsn,
        "precision" : prec_dsn,
        "recall" : recall_dsn
    }

    return lr,train_loss, dsn_train_metric, ccnet_train_metric
    