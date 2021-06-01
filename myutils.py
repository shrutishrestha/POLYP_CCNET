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
                "train_ccnet_tn", "train_ccnet_fp", "train_ccnet_fn", "train_ccnet_tp", "train_ccnet_meanIU", "train_ccnet_dice", "train_ccnet_precision", "train_ccnet_recall",
                "train_dsn_tn", "train_dsn_fp", "train_dsn_fn", "train_dsn_tp", "train_dsn_meanIU", "train_dsn_dice","train_dsn_precision","train_dsn_recall",
                "val_tn","val_fp","val_fn","val_tp","val_meanIU","val_dice","val_precision","val_recall"]])
                

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
