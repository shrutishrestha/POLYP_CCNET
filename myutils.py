import os
import csv


def check_and_make_files(filepath, result_file = False):
    if result_file:
        for file in filepath:
            if not os.path.exists(file) or os.stat(file).st_size == 0:
                with open(file, 'w') as csvfile:
                    print("new csv writen file", file)
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows([["epoch", "global_iteration", "lr",     
                "train_loss","train_ccnet_tn", "train_ccnet_fp", "train_ccnet_fn", "train_ccnet_tp", "train_ccnet_meanIU", "train_ccnet_dice", "train_ccnet_precision", "train_ccnet_recall",
                "train_dsn_tn", "train_dsn_fp", "train_dsn_fn", "train_dsn_tp", "train_dsn_meanIU", "train_dsn_dice","train_dsn_precision","train_dsn_recall",
                "val_loss","val_tn","val_fp","val_fn","val_tp","val_meanIU","val_dice","val_precision","val_recall",
                "val_loss1","val_tn_ccnet","val_fp_ccnet","val_fn_ccnet","val_tp_ccnet","val_meanIU_ccnet","val_dice_ccnet","val_precision_ccnet","val_recall_ccnet",
                "val_tn_dsn","val_fp_dsn","val_fn_dsn","val_tp_dsn","val_meanIU_dsn","val_dice_dsn","val_precision_dsn","val_recall_dsn"]])
                
                

    for file in filepath:
        if not os.path.exists(file):
            with open(file, 'w') as fp:
                print("formed new file", file)
                pass


def check_and_make_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("formed new folder", directory)
