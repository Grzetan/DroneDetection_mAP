import pandas as pd
import numpy as np
import csv
from collections import Counter
from metrics import metrics

# Box = frame_id, x1, y1, x2, y2, prob_score
def load_predicions(path: str):
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

# frame_id, x1, y1, x2, y2
def load_labels(path: str):
    data = list(csv.reader(open(path)))
    data = [[int(val) for val in row] for row in data[1:]]

    return np.array(data)

def load_predictions2(path: str):
    data = list(csv.reader(open(path)))
    boxes = []
    for row in data[1:]:
        box = [int(a) if i != 7 else float(a) for i, a in enumerate(row) if i not in [5,6]]
        box[3] = box[1] + box[3]
        box[4] = box[2] + box[4]
        boxes.append(box)

    return np.array(boxes)

def load_labels2(path: str):
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
    data = list(csv.reader(open(path), delimiter=','))
    boxes = []
    for row in data[1:]:
        for i in range((len(row)-1) // 4):
            box = [int(float(a)) for a in row[i*4+1:i*4+5]]
            box = [int(row[0]) - shift] + box
            boxes.append(box)
    return np.array(boxes)



def main():
    # preds = load_predicions("./predictions2/Dron T02.61920488.20220117151609.csv")
    # labels = load_labels("./labels2/Dron T02.61920488.20220117151609.avi.csv")

    # preds = load_predictions2("./predictions/coordinates_1.csv")
    # labels = load_labels2("./labels/connected_1.csv")

    labels = load_labels3("./labels3/Dron T02.55260362.20220117151609.avi.csv", 3)
    preds = load_predicions("./predictions3/best_640T0.2/Dron T02.55260362.20220117151609.csv")

    score = metrics(preds, labels, mAP_start=0.5, mAP_stop=0.95, mAP_step=0.05, main_iou_thresh=0.5, plot=True)
    print(score)

    # Save to CSV file
    df = pd.DataFrame([score])
    df.to_csv("output.csv")

if __name__ == "__main__": main()