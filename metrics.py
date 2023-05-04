##
# @file
# metrics file
#

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import math
import cv2

def iou(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
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

def classifyDetected(predictions: np.ndarray, labels: np.ndarray, iou_thresh) -> tuple:
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

def averagePrecision(predictions: np.ndarray, labels: np.ndarray, iou_thresh: float=0.5, plot: bool=False) -> float:
    """! Calculates average precision for given IOU threshold.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param iou_thresh Float in range (0, 1> that decides how much detected bbox should intersect with ground truth bbox to be called valid.
    @param plot If true precision vs recall graph is shown which can visualize area under curve
    @return Returns average precision score for given IOU thresh: float <0, 1>
    """

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
    for i in range(len(precisions[:-1])):
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
    # ap = simpson(precisions, recalls) # Scipy simpson function ???
    ap = sum(points) / len(points) # 11-points interpolated method

    if plot: 
        plt.title(f"Area under curve: {ap}")
        plt.suptitle(f"Graph generated for IOU threshold = {iou_thresh}")
        plt.show()

    return ap

def mAP(predictions: np.ndarray, labels: np.ndarray, iou_start: float = 0.5, iou_stop: float = 0.95, iou_step: float = 0.05, plot: bool=False) -> float:
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

    return (sum(APs) / len(APs))

def calculateCenterDist(prediction: np.ndarray, label: np.ndarray) -> float:
    """! Calculates distance between bboxe's centers.
    @param prediction Numpy array of box data. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param label Numpy array of ground truth bbox. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @return Returns distance between bboxe's centers
    """
    c1_x = label[1] + (label[3] - label[1])
    c1_y = label[2] + (label[4] - label[2])
    c2_x = prediction[1] + (prediction[3] - prediction[1])
    c2_y = prediction[2] + (prediction[4] - prediction[2])

    return math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

def distBetweenCenters(predictions: np.ndarray, labels: np.ndarray, dist_thresh: float, method: str = 'normal') -> tuple:
    """! Calculates mean distance between bboxe's centers.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param dist_thresh Float in range (0, 1> that decides how much detected bbox should intersect with ground truth bbox to be called valid.
    @param method Method of calculating distance, either 'squared' or 'normal'. Squared squares distances in euclidean distance before summing.S
    @return Returns tuple of mean distance between bboxe's centers and FN count
    """

    # First, pair label with predictions
    amount_bboxes = Counter([gt[0] for gt in labels])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = np.zeros(val)

    # Sort predictions by score
    predictions = np.array(sorted(predictions, key=lambda x: x[5], reverse=True))

    dists = []
    FN_count = 0

    # For every prediction, find groud truth bbox with highest iou and check if it is greater than iou thresh
    for pred in predictions:
        # Get gt bboxes from the same frame
        gt_bboxes = [gt for gt in labels if gt[0] == pred[0]]

        best_dist = 1e+8
        best_gt_index = 0

        for j, gt_box in enumerate(gt_bboxes):
            dist = calculateCenterDist(pred,gt_box)

            if dist < best_dist:
                best_dist = dist
                best_gt_index = j

        if best_dist < dist_thresh and amount_bboxes[pred[0]][best_gt_index] == 0:
            dists.append(best_dist)
            amount_bboxes[pred[0]][best_gt_index] = 1
        else:
            FN_count += 1

    if method == 'normal':
        return sum(dists) / len(dists), FN_count
    elif method == 'squared':
        return sum([i**2 for i in dists]) / len(dists), FN_count
    else:
        return 0, FN_count

def metrics(predictions: np.ndarray, 
            labels: np.ndarray, 
            mAP_start: float = 0.1, 
            mAP_stop: float = 0.90, 
            mAP_step: float = 0.05, 
            main_iou_thresh: float = 0.5,
            dist_thresh: int = 15,
            plot: bool=False) -> dict:
    """! Returns metrics for input data.
    @brief Tuple with 3 values is returned, first value is mAP for given mAP thresholds, second value is count of false negatives for given iou thresh, 
    last value is a mean distance between label bbox and closest prediction.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param mAP_start Mean average precision starting IOU threshold.
    @param mAP_stop Mean average precision ending IOU threshold.
    @param mAP_step Step of thresholds. Total number of checked thresholds depend on this number.
    @param main_iou_thresh IOU threshold which is used in counting FNs and calcualating distance between bboxe's centers
    @param dist_thresh Distance threshold which is responsible for calculating mean center distance
    @param plot If true precision vs recall graph is shown which can visualize area under curve
    @return Returns tuple of three: (mAP, FN_count, bbox_center_dist)
    """

    # Get mean average precision
    mAP_score = mAP(predictions, labels, mAP_start, mAP_stop, mAP_step, plot)

    # Count FN for given threshold
    TP, FP = classifyDetected(predictions, labels, main_iou_thresh)
    FN_count = len(labels) - sum(TP)

    # Get average distance between bbox centers
    dist, FP = distBetweenCenters(predictions, labels, dist_thresh)

    return {"mAP": mAP_score, "FN_count": FN_count, "mean_center_dist": dist}

def getCenterDistances(predictions: np.ndarray, labels: np.ndarray) -> list:
    """! Matches closest ground truth box to every prediction and returns list of distances between each prediction and matched ground truth box
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @return List of distances between centers.
    """

    # First, pair label with predictions
    amount_bboxes = Counter([gt[0] for gt in labels])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = np.zeros(val)

    # Sort predictions by score
    predictions = np.array(sorted(predictions, key=lambda x: x[5], reverse=True))

    dists = []

    # For every prediction, find groud truth bbox with highest iou and check if it is greater than iou thresh
    for pred in predictions:
        # Get gt bboxes from the same frame
        gt_bboxes = [gt for gt in labels if gt[0] == pred[0]]

        best_dist = 1e+8

        for j, gt_box in enumerate(gt_bboxes):
            dist = calculateCenterDist(pred,gt_box)

            if dist < best_dist:
                best_dist = dist

        if best_dist != 1e+8: dists.append(best_dist) # Not sure what to do if sample does not have any ground truths (For now, skip prediction)

    return dists

def plotMeanDistance(predictions: np.ndarray, labels: np.ndarray, start: int = 100, end: int = 2000, step: int = 10):
    """! Plots mean center distance vs FP count. Output is a opposite result to plotFNCount
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param start Starting arbitrial FN count
    @param stop Ending arbitrial FN count
    @param step Step of thresholds.
    @return void
    """

    dists = getCenterDistances(predictions, labels)

    FP_counts = []
    distances = []

    dists = sorted(dists)
    for val in np.arange(start, end, step):
        FP_counts.append(val)
        distances.append(sum(dists[:val]) / len(dists[:val]))

    plt.plot(distances, FP_counts)
    plt.xlabel('Mean Distance')
    plt.ylabel('FP count')
    # plt.title(f'Prediction count: {len(predictions)}')
    plt.show()

def plotFPCount(predictions: np.ndarray, labels: np.ndarray, start: float = 3, stop: float = 40, step: float = 1, plot: bool = True):
    """! Plots FP count vs mean center distance. False positive means that there wasa detection when no ground truth box.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param start Starting distance for bboxes to be matched
    @param stop Ending distance for bboxes to be matched
    @param step Step of thresholds. Total number of checked thresholds depend on this number.
    @param plot Controls if graph is plotted
    @return Returns FP count and list of MCD across different thresholds
    """

    FP_counts = []
    distances = []

    for val in np.arange(start, stop, step):
        dist, count = distBetweenCenters(predictions, labels, dist_thresh=val)
        FP_counts.append(count)
        distances.append(dist)
    
    if plot:
        plt.plot(FP_counts, distances)
        plt.ylabel('Mean Center Distance')
        plt.xlabel('FP count')
        # plt.title(f'Prediction count: {len(predictions)}')
        plt.show()

    return FP_counts, distances

def plotFPRate(predictions: np.ndarray, labels: np.ndarray, start: float = 3, stop: float = 40, step: float = 1):
    """! Plots FP Rate vs mean center distance. False positive means that there wasa detection when no ground truth box.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param start Starting distance for bboxes to be matched
    @param stop Ending distance for bboxes to be matched
    @param step Step of thresholds. Total number of checked thresholds depend on this number.
    @return void
    """

    dists = getCenterDistances(predictions, labels)

    FP_counts = []
    distances = []

    for val in np.arange(start, stop, step):
        valid_dists = [d for d in dists if d < val]
        FP_counts.append((len(predictions) - len(valid_dists)) / len(labels))
        distances.append(sum(valid_dists) / len(valid_dists))
    
    plt.plot(FP_counts, distances)
    plt.ylabel('Mean center distance')
    plt.xlabel('FP Rate')
    # plt.title(f'Prediction count: {len(predictions)}')
    plt.show()

def plotFNCount(predictions: np.ndarray, labels: np.ndarray, start: float = 3, stop: float = 40, step: float = 1, plot: bool = True):
    """! Plots FN Count vs mean center distance. False negative means that there wasn't a detection for ground truth box.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param start Starting distance for bboxes to be matched
    @param stop Ending distance for bboxes to be matched
    @param step Step of thresholds. Total number of checked thresholds depend on this number.
    @param plot Controls if graph is plotted
    @return FN count and list od MDC across different thresholds
    """
    
    dists = []

    # For label, find prediction bbox with highest iou and check if it is greater than iou thresh
    for label in labels:
        # Get preds from the same frame
        preds = [p for p in predictions if p[0] == label[0]]

        best_dist = 1e+8

        for j, pred in enumerate(preds):
            dist = calculateCenterDist(pred,label)

            if dist < best_dist:
                best_dist = dist

        if best_dist != 1e+8: dists.append(best_dist) # Not sure what to do here

    FN_counts = []
    distances = []

    for val in np.arange(start, stop, step):
        valid_dists = [d for d in dists if d < val]
        FN_counts.append((len(labels) - len(valid_dists)))
        distances.append(sum(valid_dists) / len(valid_dists))

    if plot:
        plt.plot(FN_counts, distances)
        plt.ylabel('Mean center distance')
        plt.xlabel('FN Count')
        # plt.title(f'Prediction count: {len(predictions)}')
        plt.show()

    return FN_counts, distances

def plotFNRate(predictions: np.ndarray, labels: np.ndarray, start: float = 3, stop: float = 40, step: float = 1):
    """! Plots FN Rate vs mean center distance. False negative means that there wasn't a detection for ground truth box.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param start Starting distance for bboxes to be matched
    @param stop Ending distance for bboxes to be matched
    @param step Step of thresholds. Total number of checked thresholds depend on this number.
    @return void
    """
    
    dists = []

    # For label, find prediction bbox with highest iou and check if it is greater than iou thresh
    for label in labels:
        # Get preds from the same frame
        preds = [p for p in predictions if p[0] == label[0]]

        best_dist = 1e+8

        for j, pred in enumerate(preds):
            dist = calculateCenterDist(pred,label)

            if dist < best_dist:
                best_dist = dist

        if best_dist != 1e+8: dists.append(best_dist) # Not sure what to do here

    FN_counts = []
    distances = []

    for val in np.arange(start, stop, step):
        valid_dists = [d for d in dists if d < val]
        FN_counts.append((len(labels) - len(valid_dists)) / len(labels))
        distances.append(sum(valid_dists) / len(valid_dists))
    
    plt.plot(FN_counts, distances)
    plt.ylabel('Mean center distance')
    plt.xlabel('FN Rate')
    # plt.title(f'Prediction count: {len(predictions)}')
    plt.show()

def calculateRates(predictions: np.ndarray, labels: np.ndarray, thresh: int):
    """! Calculates rates (mean center distance, false positive rate, false nagative rate) for given threshold.
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param thresh Threshold used in matching process
    @return map of {"MCD", "FNR", "FPR"}.
    """

    map = {}

    # First get FN rate
    dists = []

    # For label, find prediction bbox with highest iou and check if it is greater than iou thresh
    for label in labels:
        # Get preds from the same frame
        preds = [p for p in predictions if p[0] == label[0]]

        best_dist = 1e+8

        for j, pred in enumerate(preds):
            dist = calculateCenterDist(pred,label)

            if dist < best_dist:
                best_dist = dist

        if best_dist != 1e+8: dists.append(best_dist) # Not sure what to do here

    valid_dists = [d for d in dists if d < thresh]
    map['FNR'] = ((len(labels) - len(valid_dists)) / len(labels))

    # Now calcualte FPR
    dists = getCenterDistances(predictions, labels)
    valid_dists = [d for d in dists if d < thresh]
    map['FPR'] = ((len(predictions) - len(valid_dists)) / len(labels))
    map['MCD'] = (sum(valid_dists) / len(valid_dists))

    return map

def visualizeDataset(predictions: np.ndarray, labels: np.ndarray, video: str = "", start: int = 0, step: int = 10, n: int = 10):
    """! Visualizes dataset using openCV
    @param predictions Numpy array of detected bboxes. Bbox format: [frame_id, x1, y1, x2, y2, score]: ndarray
    @param labels Numpy array of ground truth bboxes. Bbox format: [frame_id, x1, y1, x2, y2]: ndarray
    @param Video Optional path to video so background is an actual frame from video not just a black background.
    @param start Starting frame to visualize
    @param step Jump between frames.
    @param n How many frames to visualize
    @return void
    """
    height = 1082
    width = 1924
    scale = 1

    vid = None
    if video != "":
        vid = cv2.VideoCapture(video)

    for i in range(start, start+n*step, step):
        image = np.zeros((height,width,3), np.uint8)

        if vid is not None:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, image = vid.read()

        label = labels[i]
        image = cv2.rectangle(image, (int(label[1] * scale), int(label[2] * scale)), (int(label[3] * scale), int(label[4] * scale)), (0,255,0), 1)

        # Load predictions
        ps = [p for p in predictions if p[0] == label[0]]
        for p in ps:
            image = cv2.rectangle(image, (int(p[1] * scale), int(p[2] * scale)), (int(p[3] * scale), int(p[4] * scale)), (0,0, 255), 1)
            cv2.putText(image, f"{round(iou(np.array(p[1:-1]), np.array(label[1:])), 2)}", (int(p[3]*scale), int(p[2]*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(f"Visualization", image)
        cv2.waitKey(0)