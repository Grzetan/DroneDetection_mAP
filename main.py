import pandas as pd
import numpy as np
import csv
import cv2
import random
import os
from collections import Counter
from metrics import metrics, iou

def load_predicions(path: str):
    """! Loads predictions from CSV file where each row contains all detected drones (row length is dependant on count of detected drones)
        @returns Numpy array where each record = [frame_id, x1, y1, x2, y2, score]
    """
    data = list(csv.reader(open(path)))
    boxes = []
    for row in data:
        n_boxes = (len(row) - 1) // 7
        for k in range(n_boxes):
            box = [float(a) for j, a in enumerate(row[k*7+1:k*7+8]) if j not in [4, 5]]
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append([int(row[0])] + box)
    return np.array(boxes)

def load_labels(path: str):
    """! Loads labels from CSV file where each row contains single record [frame_id, x1, y1, x2, y2]
        @returns Numpy array where each record = [frame_id, x1, y1, x2, y2]
    """
    data = list(csv.reader(open(path)))
    data = [[int(val) for val in row] for row in data]

    return np.array(data)

def load_predictions2(path: str):
    """! Loads predictions from CSV file where each row contains single record [frame_id, x, y, w, h, prob]
        @returns Numpy array where each record = [frame_id, x1, y1, x2, y2, prob]
    """
    data = list(csv.reader(open(path)))
    boxes = []
    for row in data[1:]:
        box = [int(a) if i != 7 else float(a) for i, a in enumerate(row) if i not in [5,6]]
        box[3] = box[1] + box[3]
        box[4] = box[2] + box[4]
        boxes.append(box)

    return np.array(boxes)

def load_labels2(path: str):
    """! Loads labels from CSV file where each row contains data for 0, one or two drones. Each drone is [x, y, w, h]. First numer in row is frame_id.
        @returns Numpy array where each record = [frame_id, x1, y1, x2, y2]
    """
    data = list(csv.reader(open(path), delimiter=';'))
    boxes = []
    for row in data:
        for i in range((len(row)-1) // 4):
            box = [int(float(a)) for a in row[i*4+1:i*4+5]]
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            box = [int(row[0])] + box
            boxes.append(box)
    return np.array(boxes)

def load_labels3(path: str, shift: int):
    """! Loads labels from CSV file where each row contains single record [frame_id, x, y, w, h]. It also has the frame shift option
        @returns Numpy array where each record = [frame_id, x1, y1, x2, y2]
    """
    data = list(csv.reader(open(path), delimiter=','))
    boxes = []
    for row in data[1:]:
        for i in range((len(row)-1) // 4):
            box = [int(float(a)) for a in row[i*4+1:i*4+5]]
            box = [int(row[0]) - shift] + box
            boxes.append(box)
    return np.array(boxes)

def visualizeDataset(preds: np.ndarray, labels: np.ndarray, scale: float = 0.5):
    height = 800
    width = 1000
    scale = 0.5

    random.shuffle(labels)

    for i, label in enumerate(labels[-5:]):
        image = np.zeros((height,width,3), np.uint8)
        # Label
        image = cv2.rectangle(image, (int(label[1] * scale), int(label[2] * scale)), (int(label[3] * scale), int(label[4] * scale)), (0,255,0), 1)

        # Load predictions
        ps = [p for p in preds if p[0] == label[0]]
        for p in ps:
            image = cv2.rectangle(image, (int(p[1] * scale), int(p[2] * scale)), (int(p[3] * scale), int(p[4] * scale)), (0,0, 255), 1)
            cv2.putText(image, f"{round(iou(np.array(p[1:-1]), np.array(label[1:])), 2)}", (int(p[3]*scale), int(p[2]*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(f"{label[0]}", image)
        cv2.waitKey(0)

def main():
    # preds = load_predicions("./predictions2/Dron T02.61920488.20220117151609.csv")
    # labels = load_labels("./labels2/Dron T02.61920488.20220117151609.avi.csv")

    # preds = load_predictions2("./predictions/coordinates_1.csv")
    # labels = load_labels2("./labels/connected_1.csv")


    # New data

    # label_dir = "./labels4_lab_15"
    # predictions_dir = "./AirSim/lab_15"

    # models = [p for p in os.listdir(predictions_dir) if p.startswith('airSim') and not p.endswith('onnx')]

    # all_metrics = []

    # for model in models:
    #     print(model)
    #     cams = sorted([p for p in os.listdir(os.path.join(predictions_dir, model)) if p.endswith('.csv')])
    #     mAPs = []
    #     FN_counts = []
    #     mean_distances = []
    #     for i, cam in enumerate(cams):
    #         print(cam)
    #         labels = load_labels(os.path.join(label_dir, f'labelsCam{i+1}.csv'))
    #         preds = load_predicions(os.path.join(predictions_dir, model, cam))

    #         result = metrics(preds, labels)
    #         mAPs.append(result['mAP'])
    #         FN_counts.append(result['FN_count'])
    #         mean_distances.append(result['mean_center_dist'])
        
    #     all_metrics.append([model, sum(mAPs) / len(mAPs), sum(FN_counts), sum(mean_distances) / len(mean_distances)])

    # df = pd.DataFrame(all_metrics, columns=['Model', 'mAP', 'FN_count', 'mean_center_distance'])
    # df.to_csv('metrics_lab15.csv')

    # Old data

    label_dir = "./labels3"
    predictions_dir = "./predictions3"

    labels_files = [l for l in os.listdir(label_dir) if l.endswith('.avi.csv')]
    
    all_metrics = []

    for path in os.listdir(predictions_dir):
        files = os.listdir(os.path.join(predictions_dir, path))
        mAPs = []
        FN_counts = []
        mean_distances = []

        for l in labels_files:
            pred_file = [f for f in files if f.split('.csv')[0] == l.split('.avi.csv')[0]][0]
            print(os.path.join(label_dir, l), os.path.join(predictions_dir, path, pred_file))

            preds = load_predicions(os.path.join(predictions_dir, path, pred_file))
            labels = load_labels3(os.path.join(label_dir, l), 4)

            result = metrics(preds, labels, mAP_start=0.5, mAP_stop=0.9, mAP_step=0.05, main_iou_thresh=0.5, plot=False)

            mAPs.append(result['mAP'])
            FN_counts.append(result['FN_count'])
            mean_distances.append(result['mean_center_dist'])

        all_metrics.append([path, sum(mAPs) / len(mAPs), sum(FN_counts), sum(mean_distances) / len(mean_distances)])

    df = pd.DataFrame(all_metrics, columns=['Model', 'mAP', 'FN_count', 'mean_center_distance'])
    df.to_csv('output.csv')



    # Visualize dataset

    # labels = load_labels3("./labels3/Dron T02.55260362.20220117151609.avi.csv", 3)
    # preds = load_predicions("./predictions3/best_640T0.2/Dron T02.55260362.20220117151609.csv")

    # visualizeDataset(preds, labels)

    # score = metrics(preds, labels, mAP_start=0.5, mAP_stop=0.95, mAP_step=0.05, main_iou_thresh=0.5, plot=True)
    # print(score)

    # # Save to CSV file
    # df = pd.DataFrame([score])
    # df.to_csv("output.csv")

if __name__ == "__main__": main()