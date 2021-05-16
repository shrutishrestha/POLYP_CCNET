import numpy as np
import torch

def get_confusion_matrix(gt_label, pred_label, class_num): #seg_gt, seg_pred, args.num_classes
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """

        pred_label = pred_label.flatten()
        if torch.is_tensor(gt_label) == True:
            gt_label = gt_label.cpu().detach().numpy()

        gt_label = gt_label.flatten()

        index = (gt_label * class_num + pred_label).astype('int32') #gt_label(array([0, 1]), array([316446,  12684])) pred_label (array([0, 1], dtype=uint8), array([ 77728, 251402]))

        label_count = np.bincount(index)

        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def calculate_metrics(confusion_matrix):
    tn, fp, fn, tp = expand_confusion_matrix(confusion_matrix)
    dice = dice_coef(tn, fp, fn, tp)
    meanIU = mean_IU(confusion_matrix)
    prec = precision(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return tn, fp, fn, tp, meanIU, dice, prec, rec


def precision(tn, fp, fn, tp):
    return tp/(tp+fp)

def recall(tn, fp, fn, tp):
    return tp/(tp+fn)


def expand_confusion_matrix(confusion_matrix):
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    return tn, fp, fn, tp


def dice_coef(tn, fp, fn, tp, smooth=0.000001, activation = 'sigmoid'):
    if activation is None or activation == 'none':
        activation_fn = lambda x:x

    dice = 2*tp / ((tp+fp)+(tp+fn)+smooth)
    return dice


def mean_IU(confusion_matrix):
    pos = confusion_matrix.sum(1) # actual no actual yes
    res = confusion_matrix.sum(0) # pred no pred yes
    tp = np.diag(confusion_matrix)

    precision = tp / res[1]
    recall = tp / pos[1]

    IU_array = (tp / np.maximum(1.0, pos + res - tp))

    mean_IU = IU_array.mean()
    return mean_IU



