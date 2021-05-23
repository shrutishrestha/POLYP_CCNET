import argparse
import csv
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import json
import torch.optim as optim
import scipy.misc
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import sys
import torchvision
import os
import shutil
from tqdm import tqdm
import os.path as osp
import networks
from dataset.datasets import KvasirSegDataSet
from networks.ccnet import ResNet, Bottleneck, RCCAModule
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.nn import functional as F
from networks.unet import UNet
from utils.pyt_utils import load_model
import random
import time
import logging
from utils.image_utils import get_train_merged_image, get_val_merged_image
from utils.pyt_utils import decode_labels, inv_preprocess, decode_predictions
from loss.criterion import CriterionDSN, CriterionOhemDSN, CriterionOhemDSN2
from engine import Engine
from genericpath import exists
from evaluate import validation_method
from train import train_method
from write import write_in_tensorboard, write_in_csv
from myutils import check_and_make_directories, check_and_make_files


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser(config):
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--train-data-path", type=str, default=config['training']['train-data-path'],
                        help="training data path")

    parser.add_argument("--power", type=float, default=config['training']['power'],
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=config['training']['train-batch-size'],
                        help="Number of train images sent to the network in one step.")
    parser.add_argument("--momentum", type=float, default=config['training']['momentum'],
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=config['training']['num-classes'],
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=config['training']['ignore-label'],
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--random-seed", type=int, default=config['training']["random-seed"],
                        help="Random seed to have reproducible results.")
    parser.add_argument("--learning-rate", type=float, default=config['training']['learning-rate'],
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--weight-decay", type=float, default=config['training']['weight-decay'],
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--input-size", type=str, default=config['training']['input-size'],
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--snapshot-dir", type=str, default=config['training']['snapshot-dir'],
                        help="Where to save snapshots of the model.")
    parser.add_argument("--best-checkpoint-fpath", type=str, default=config['training']['best-checkpoint-fpath'],
                        help="best checkpoint file path")
    parser.add_argument("--current-checkpoint-fpath", type=str, default=config['training']['current-checkpoint-fpath'],
                        help="current checkpoint file path")
    parser.add_argument("--restore-from", type=str, default=config['training']['restore-from'],
                        help="Where restore model parameters from.")
    parser.add_argument("--start-iters", type=int, default=config['training']['start-iters'],
                        help="Number of classes to predict (including background).")

    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--gpu", type=str, default=config['training']['gpu'],
                        help="choose gpu device.")
    parser.add_argument("--model", type=str, default=config['training']['model'],
                        help="choose model.")
    parser.add_argument("--recurrence", type=int, default=config['training']['recurrence'],
                        help="choose the number of recurrence.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--result-file-path", type=str, default=config['training']['result-file-path'],
                        help="path to store loss values")
    parser.add_argument("--result-dir", type=str, default=config['training']['result-dir'],
                        help="folder path")
    parser.add_argument("--num-workers", type=int, default=config['training']['num-workers'],
                        help="choose the number of workers.")
    parser.add_argument("--tensorboard-output", type=str, default=config['training']['tensorboard-output'],
                        help="tensorboard-output path to save image")
    parser.add_argument("--max-epochs", type=int, default=config['training']['max-epochs'],
                        help="maximum epochs")
    parser.add_argument("--ohem", type=bool, default=True,
                        help="maximum epochs")
    parser.add_argument("--ohem-thres", type=float, default=0.7,
                        help="ohem threshold")
    parser.add_argument("--ohem-keep", type=float, default=100000,
                        help="ohem keep value")


    parser.add_argument("--test-data-path", type=str, default=config['evaluation']['test-data-path'],
                        help="testing data path")
    parser.add_argument("--Output", type=str, default="/Users/shruti/NAAMIIProjects/CCNet-Polyp-data_aug",
                        help="testing data path")
    parser.add_argument("--test-batch-size", type=int, default=config['training']['test-batch-size'],
                        help="Number of test images sent to the network in one step.")
    return parser




def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main(config):
    """Create the model and start the training."""
    parser = get_parser(config)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        summary_writer = SummaryWriter(args.tensorboard_output)

        cudnn.benchmark = True

        # fix a seed for reproducibility
        seed = args.random_seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # for dataset and dataloader
        test_loader, train_loader, train_sampler = get_data_loaders(args, engine)

        # for model and criterion
        if args.ohem:
            criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
        else:
            criterion = CriterionDSN() 

        seg_model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes, criterion=criterion,
                           recurrence=args.recurrence)
        seg_model = load_model(seg_model, model_file=args.restore_from)

        optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}],
            lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        # initializing models to cuda
        seg_model.to(device)
        model = torch.nn.DataParallel(seg_model)

        # check and making directories
        check_and_make_directories([args.snapshot_dir, args.result_dir])
        check_and_make_files([args.result_file_path], result_file=True)
        check_and_make_files([args.current_checkpoint_fpath, args.best_checkpoint_fpath])


        run = True
        epoch = args.start_iters
        global_iteration = epoch * len(train_loader)

        csv_filepath = args.result_file_path
        best_val_metric = 0

        while run:
            print("epoch",epoch)
            if epoch >= args.max_epochs:
                run = False
                break

            lr, train_loss, train_dsn_metric, train_ccnet_metric = train_method(
                epoch = epoch,
                args = args,
                criterion = criterion,
                engine = engine,
                global_iteration = global_iteration,
                model = model,
                optimizer = optimizer,
                train_loader = train_loader,
                train_sampler = train_sampler,
                summary_writer = summary_writer
            )

            val_loss, val_metric = validation_method(
                epoch = epoch,
                args = args, 
                model = model, 
                test_loader = test_loader,
                criterion = criterion,
                summary_writer = summary_writer
                )    

            torch.save(seg_model.state_dict(), args.current_checkpoint_fpath)

            if val_metric["dice"] > best_val_metric:
                best_val_metric = val_metric["dice"]
                shutil.copyfile(args.current_checkpoint_fpath, args.best_checkpoint_fpath)
            
            epoch +=1 

            write_in_tensorboard(
            epoch = epoch,
            summary_writer = summary_writer,
            train_ccnet_metric = train_ccnet_metric,
            train_dsn_metric = train_dsn_metric,
            train_loss = train_loss,
            val_loss = val_loss,
            val_metric = val_metric)

            write_in_csv(
            filename = csv_filepath,
            epoch = epoch,
            global_iteration = global_iteration,
            lr = lr,
            train_loss = train_loss,
            train_ccnet_metric = train_ccnet_metric,
            train_dsn_metric = train_dsn_metric,
            val_loss = val_loss,
            val_metric = val_metric
            )


def get_data_loaders(args, engine):
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    traindataset = KvasirSegDataSet(data_dir=args.train_data_path, max_iters=None, crop_size=input_size,
                                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                    ignore_label=args.ignore_label, test=False)  
    testdataset = KvasirSegDataSet(data_dir=args.test_data_path, crop_size=input_size, mean=IMG_MEAN, scale=False, mirror=False, test=True)
    # for loaders
    train_loader, train_sampler = engine.get_train_loader(traindataset)
    test_loader, test_sampler = engine.get_test_loader(testdataset)

    return test_loader, train_loader, train_sampler


if __name__ == '__main__':
    config = json.load(open("./config.json"))
    main(config)
