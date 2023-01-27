import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def iou(pred: np.ndarray, label: np.ndarray):
    """! Calculates intersection over union of two input boxes
    @param pred First box. Bbox format: [x1, y1, x2, y2]: ndarray
    @param label Second box. Bbox format: [x1, y1, x2, y2]: ndarray
    @return Returns iou score for boxes <0, 1>: float
    """
    x1 = max(pred[0], label[0])
    y1 = max(pred[1], label[1])
    x2 = min(pred[2], label[2])
    y2 = min(pred[3], label[3])

    intersection = max((x2 - x1), 0) * max((y2 - y1), 0)
    box1_area = abs((pred[2] - pred[0]) * (pred[3] - pred[1]))
    box2_area = abs((label[2] - label[0]) * (label[3] - label[1]))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def classifyDetected(predictions: np.ndarray, labels: np.ndarray, iou_thresh):
    """! Classifies detected bboxes to true positive or false positive.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param iou_thresh Float in range (0, 1> that decides how much detected bbox should intersect with ground truth bbox to be called valid.
    @return Returns tuple of two numpy arrays, first one is binary mask of true positives and second one is binary mask for false positives
    """

    # Dict which has information about how many true bboxes are on each frame
    amount_bboxes = Counter([gt[0] for gt in labels])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = np.zeros(val)

    TP = np.zeros((len(predictions)), dtype=np.uint8)
    FP = np.zeros((len(predictions)), dtype=np.uint8)

    # Sort predictions by score
    predictions = np.array(sorted(predictions, key=lambda x: x[5], reverse=True))

    # For every prediction, find groud truth bbox with highest iou and check if it is greater than iou thresh
    for i, pred in enumerate(predictions):
        # Get gt bboxes from the same frame
        gt_bboxes = [gt for gt in labels if gt[0] == pred[0]]

        best_iou = 0
        best_gt_index = 0

        for j, gt_box in enumerate(gt_bboxes):
            iou_score = iou(pred[1:-1], gt_box[1:])

            if iou_score > best_iou:
                best_iou = iou_score
                best_gt_index = j

        if best_iou > iou_thresh:
            if amount_bboxes[pred[0]][best_gt_index] == 0:
                TP[i] = 1
                amount_bboxes[pred[0]][best_gt_index] = 1
            else:
                FP[i] = 1
        else:
            FP[i] = 1

    return TP, FP

def averagePrecision(predictions: np.ndarray, labels: np.ndarray, iou_thresh: int=0.5, plot: bool=False):
    """! Calculates average precision for given IOU threshold.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param iou_thresh Float in range (0, 1> that decides how much detected bbox should intersect with ground truth bbox to be called valid.
    @param plot If true precision vs recall graph is shown which can visualize area under curve
    @return Returns average precision score for given IOU thresh: float <0, 1>
    """

    # Only around 541 of 24900 all predicted boxes are TP?????

    TP, FP = classifyDetected(predictions, labels, iou_thresh)

    # Calculate cumulative sum of TP and FP to than use them to create plot of precision vs recall 
    TPcumsum = np.cumsum(TP)
    FPcumsum = np.cumsum(FP)
    recalls = TPcumsum / len(labels)
    precisions = TPcumsum / (TPcumsum + FPcumsum)

    # Add point (0,1) to the beggining and (1, 0) to the end to ensure that area under curve is calculated properly
    recalls = np.insert(recalls, 0, 0, axis=0)
    precisions = np.insert(precisions, 0, 1, axis=0)
    recalls = np.append(recalls, 1)
    precisions = np.append(precisions, 0)

    # Smooth precision curve
    for i, p in enumerate(precisions[:-1]):
        precisions[i] = max(precisions[i+1:])

    if plot:
        plt.plot(recalls, precisions)
        plt.xlabel('recall')
        plt.ylabel('precision')

    # Use 11-points interpolated method
    dx = np.linspace(0, 1, 11)
    points = []
    for val in dx:
        idx = (np.abs(recalls - val)).argmin()
        points.append(precisions[idx])
        if plot:
            plt.plot(val, precisions[idx], 'ro')

    # Calculate area under precision-recall curve
    # ap = np.trapz(precisions, recalls) # Numpy trapz function
    # ap = simpson(precisions, recalls) # Scipy simpson function
    ap = sum(points) / len(points) # 11-points interpolated method

    if plot: 
        plt.title(f"Area under curve: {ap}")
        plt.suptitle(f"Graph generated for IOU threshold = {iou_thresh}")
        plt.show()

    return ap

def mAP(predictions: np.ndarray, labels: np.ndarray, iou_start: float, iou_stop: float, iou_step: float, plot: bool=False):
    """! Calculates mean average precision for every value in given IOU range.
    @brief For iou_start=0.5, iou_stop=0.9, iou_step=0.1 these IOU thresholds will be checked: [0.5, 0.6, 0.7, 0.8, 0.9].
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param iou_start Starting IOU threshold.
    @param iou_stop Ending IOU threshold.
    @param iou_step Step of thresholds. Total number of checked thresholds depend on this number.
    @param plot If true precision vs recall graph is shown which can visualize area under curve
    @return Returns mean average precision score for given IOU range: float <0, 1>
    """

    APs = []
    for thresh in np.arange(iou_start, iou_stop, iou_step):
        APs.append(averagePrecision(predictions, labels, thresh, plot))

    return sum(APs) / len(APs)