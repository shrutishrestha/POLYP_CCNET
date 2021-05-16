from torch.utils.tensorboard import SummaryWriter
import csv


def write_in_tensorboard(epoch,
    summary_writer, 
    train_ccnet_metric,
    train_dsn_metric ,
    train_loss ,
    val_loss,
    val_metric ):
    
    train_ccnet_dice = train_ccnet_metric["dice"] 
    train_dsn_dice = train_dsn_metric["dice"]
    val_dice = val_metric["dice"]

    summary_writer.add_scalar("training/train_loss", train_loss, epoch)
    summary_writer.add_scalar("training/train_ccnet_metric", train_ccnet_dice, epoch)
    summary_writer.add_scalar("training/train_dsn_metric", train_dsn_dice, epoch)
    summary_writer.add_scalar("validation/val_loss", val_loss, epoch)
    summary_writer.add_scalar("validation/val_dice", val_dice, epoch)


def write_in_csv(
    filename, 
    epoch, 
    global_iteration, 
    lr, 
    train_loss, 
    train_ccnet_metric,
    train_dsn_metric, 
    val_loss, 
    val_metric):

    train_ccnet_tn = train_ccnet_metric["tn"] 
    train_ccnet_fp = train_ccnet_metric["fp"] 
    train_ccnet_fn = train_ccnet_metric["fn"] 
    train_ccnet_tp = train_ccnet_metric["tp"] 
    train_ccnet_meanIU = train_ccnet_metric["meanIU"] 
    train_ccnet_dice = train_ccnet_metric["dice"] 
    train_ccnet_precision = train_ccnet_metric["precision"] 
    train_ccnet_recall = train_ccnet_metric["recall"] 

    train_dsn_tn = train_dsn_metric["tn"] 
    train_dsn_fp = train_dsn_metric["fp"] 
    train_dsn_fn = train_dsn_metric["fn"] 
    train_dsn_tp = train_dsn_metric["tp"] 
    train_dsn_meanIU = train_dsn_metric["meanIU"] 
    train_dsn_dice = train_dsn_metric["dice"] 
    train_dsn_precision= train_dsn_metric["precision"] 
    train_dsn_recall = train_dsn_metric["recall"] 

    val_tn = val_metric["tn"]
    val_fp = val_metric["fp"]
    val_fn = val_metric["fn"]
    val_tp = val_metric["tp"]
    val_meanIU = val_metric["meanIU"]
    val_dice = val_metric["dice"]
    val_precision = val_metric["precision"]
    val_recall = val_metric["recall"]

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([[epoch, global_iteration, round(lr, 7),     
        train_ccnet_tn, train_ccnet_fp, train_ccnet_fn, train_ccnet_tp,
        round(train_ccnet_meanIU, 7),
        round(train_ccnet_dice, 7),
        round(train_ccnet_precision, 7),
        round(train_ccnet_recall, 7),
        train_dsn_tn, train_dsn_fp, train_dsn_fn, train_dsn_tp,
        round(train_dsn_meanIU, 7),
        round(train_dsn_dice, 7),
        round(train_dsn_precision, 7),
        round(train_dsn_recall, 7),
        val_tn,val_fp,val_fn,val_tp,
        round(val_meanIU, 7),
        round(val_dice, 7),
        round(val_precision, 7),
        round(val_recall, 7)]])



