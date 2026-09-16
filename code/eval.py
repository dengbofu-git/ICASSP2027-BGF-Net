import numpy as np

from utils.eval_functions import *


def eval_for_testAllInOne(opt, pred, gt):
    pred_mask = np.array(pred)
    gt_mask = np.array(gt)

    Thresholds = np.linspace(1, 0, 256)

    assert pred_mask.shape == gt_mask.shape

    gt_mask = (gt_mask > 0.5).astype(np.float64)
    pred_mask = pred_mask.astype(np.float64) / 255

    Smeasure = StructureMeasure(pred_mask, gt_mask)
    wFmeasure = original_WFb(pred_mask, gt_mask)
    MAE = np.mean(np.abs(gt_mask - pred_mask))

    threshold_E = np.zeros(len(Thresholds))
    threshold_F = np.zeros(len(Thresholds))
    threshold_Pr = np.zeros(len(Thresholds))
    threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))

    for j, threshold in enumerate(Thresholds):
        threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)
        Bi_pred = np.zeros_like(pred_mask)
        Bi_pred[pred_mask >= threshold] = 1
        threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

    meanDic = np.mean(threshold_Dic)
    meanIoU = np.mean(threshold_Iou)
    meanEm = np.mean(threshold_E)
    maxEm = np.max(threshold_E)

    mae = np.mean(MAE)
    Sm = np.mean(Smeasure)
    wFm = np.mean(wFmeasure)

    results = []
    for metric in opt["metrics"]:
        results.append(eval(metric))
    return results
