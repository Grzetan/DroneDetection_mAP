import pandas as pd
import numpy as np
import csv
from collections import Counter

# Box = frame_id, x1, y1, x2, y2, prob_score
def load_predicions(path: str):
    data = list(csv.reader(open(path)))
    boxes = []
    for i, row in enumerate(data):
        n_boxes = (len(row) - 1) // 7
        for i in range(n_boxes):
            box = [float(a) for j, a in enumerate(row[i*7+1:i*7+8]) if j not in [4, 5]]
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append([int(row[0])] + box)
    return boxes

# frame_id, x1, y1, x2, y2
def load_labels(path: str):
    data = list(csv.reader(open(path)))
    data = [[int(val) for val in row] for row in data[1:]]

    return data

def iou_multiple_boxes(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def iou_single(pred, label):
    x1 = max(pred[0], label[0])
    y1 = max(pred[1], label[1])
    x2 = min(pred[2], label[2])
    y2 = min(pred[3], label[3])

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = max((x2 - x1), 0) * max((y2 - y1), 0)
    box1_area = abs((pred[2] - pred[0]) * (pred[3] - pred[1]))
    box2_area = abs((label[2] - label[0]) * (label[3] - label[1]))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

# Pred boxes - frame_id, x1, y1, x2, y2, prob_score
# true boxes - frame_id, x1, y1, x2, y2
def mean_average_precision(
    predictions, labels, iou_threshold=0.5
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # box1 = np.array(labels[:5])[:, 1:]
    # box2 = np.array(labels[:5])[:, 1:]
    # box1 = np.array([[0,0,20,20], [0,0,10,10], [20,20,40,40]])
    # box2 = np.array([[0,0,10,10], [10,10,20,20], [20,20,40,40]])
    # print(intersection_over_union(box1, box2))

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in labels])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = np.zeros(val)

    # sort by box probabilities
    predictions.sort(key=lambda x: x[5], reverse=True)
    TP = np.zeros((len(predictions)))
    FP = np.zeros((len(predictions)))
    total_true_bboxes = len(labels)
    
    # If none exists, skip
    if total_true_bboxes == 0:
        return 0

    for detection_idx, detection in enumerate(predictions):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in labels if bbox[0] == detection[0]
        ]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = iou_single(
                np.array(detection[1:-1]),
                np.array(gt[1:])
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = np.cumsum(TP, axis=0)
    FP_cumsum = np.cumsum(FP, axis=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = [1] + precisions
    recalls = [0] + recalls

    return np.trapz(precisions, recalls)

def main():
    preds = load_predicions("./predictions2/Dron T02.55260362.20220117151609.csv")
    labels = load_labels("./labels2/Dron T02.55260362.20220117151609.avi.csv")

    mAPs = []
    for thresh in np.arange(0.2, 0.95, 0.05):
        mAPs.append(mean_average_precision(preds, labels, thresh))

    print(mAPs)
    print(sum(mAPs) / len(mAPs))

main()
