import numpy as np
import csv
import os
import pandas as pd
from metrics import *

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

def plotAirSimData(prediction_path: str, label_path: str):
    models = [p for p in os.listdir(prediction_path) if p.startswith('airSim') and not p.endswith('onnx')]

    models = sorted(models)

    for model in [models[1]]:
        # These lists will be used to plot graphs
        FN_counts_model = []
        FP_counts_model = []
        FN_MDC_model = []
        FP_MDC_model = []

        print(model)
        cams = sorted([p for p in os.listdir(os.path.join(prediction_path, model)) if p.endswith('.csv')])
        FN_counts_cameras = []
        FN_MDC_cameras = []
        FP_counts_cameras = []
        FP_MDC_cameras = []

        for i, cam in enumerate(cams):
            print(cam)
            labels = load_labels(os.path.join(label_path, f'labelsCam{i+1}.csv'))
            preds = load_predicions(os.path.join(prediction_path, model, cam))

            fn, fn_mcd = plotFNCount(preds, labels, plot=False)
            fp, fp_mcd = plotFPCount(preds, labels, plot=False)

            # If we want rates instead of counts
            fn = [f / len(labels) for f in fn]
            fp = [f / len(labels) for f in fp]

            FN_counts_cameras.append(fn)
            FP_counts_cameras.append(fp)
            FN_MDC_cameras.append(fn_mcd)
            FP_MDC_cameras.append(fp_mcd)
        
        # Average results from all cameras element wise
        for i in range(len(FN_counts_cameras[0])):
            fn_vals = [val[i] for val in FN_counts_cameras]
            fn_mcd_vals = [dis[i] for dis in FN_MDC_cameras]
            fp_vals = [val[i] for val in FP_counts_cameras]
            fp_mcd_vals = [dis[i] for dis in FP_MDC_cameras]

            FN_counts_model.append(sum(fn_vals) / len(fn_vals))
            FN_MDC_model.append(sum(fn_mcd_vals) / len(fn_mcd_vals))
            FP_counts_model.append(sum(fp_vals) / len(fp_vals))
            FP_MDC_model.append(sum(fp_mcd_vals) / len(fp_mcd_vals))

        plt.plot(FN_counts_model, FN_MDC_model)
        plt.ylabel('Mean center distance')
        plt.xlabel('FN Rate')
        plt.title(f'Model: {model}')
        plt.show()

        plt.plot(FP_counts_model, FP_MDC_model)
        plt.ylabel('Mean center distance')
        plt.xlabel('FP Rate')
        plt.title(f'Model: {model}')
        plt.show()

def getHMLMetrics(label_dir, predictions_dir):
    labels_files = [l for l in os.listdir(label_dir) if l.endswith('.avi.csv')]
    
    all_metrics = []

    for thresh in [10, 20, 30]:
        for path in os.listdir(predictions_dir):
            files = os.listdir(os.path.join(predictions_dir, path))
            MCD = []
            FNR = []
            FPR = []

            for l in labels_files:
                pred_file = [f for f in files if f.split('.csv')[0] == l.split('.avi.csv')[0]][0]
                print(os.path.join(label_dir, l), os.path.join(predictions_dir, path, pred_file))

                preds = load_predicions(os.path.join(predictions_dir, path, pred_file))
                labels = load_labels3(os.path.join(label_dir, l), 4)

                result = calculateRates(preds, labels, thresh = thresh)
                MCD.append(result['MCD'])
                FNR.append(result['FNR'])
                FPR.append(result['FPR'])

            all_metrics.append([path + f"-THRESH-{thresh}", sum(MCD) / len(MCD), sum(FNR) / len(FNR), sum(FPR) / len(FPR)])

    df = pd.DataFrame(all_metrics, columns=['Model', 'MCD', 'FNR', 'FPR'])
    df.to_csv('HML_rates.csv')

def main():
    # preds = load_predicions("./predictions2/Dron T02.61920488.20220117151609.csv")
    # labels = load_labels("./labels2/Dron T02.61920488.20220117151609.avi.csv")

    # preds = load_predictions2("./predictions/coordinates_1.csv")
    # labels = load_labels2("./labels/connected_1.csv")

    # New data

    # label_dir = "./labels/labels4_lab_12"
    # predictions_dir = "./AirSim/lab_12"

    # models = [p for p in os.listdir(predictions_dir) if p.startswith('airSim') and not p.endswith('onnx')]

    # all_metrics = []
    # models = sorted(models)

    # for model in [models[1]]:
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
    # df.to_csv('metrics_lab12.csv')
    
    # Old data

    getHMLMetrics('./labels/labels3', './predictions/predictions3')

    # plot lab sequences
    # plotAirSimData('./AirSim/lab_12', './labels/labels4_lab_12')

    # labels = load_labels3("./labels/labels3/Dron T02.55260362.20220117151609.avi.csv", 3)
    # preds = load_predicions("./predictions/predictions3/best_640T0.2/Dron T02.55260362.20220117151609.csv")

    # visualizeDataset(preds, labels, video='./predictions3/best_640T0.2/Dron T02.55260362.20220117151609.mp4')

    # score = metrics(preds, labels, mAP_start=0.5, mAP_stop=0.55, mAP_step=0.05, main_iou_thresh=0.5, dist_thresh=20, plot=False)
    # print(score)

    # plotMeanDistance(preds, labels, end=int(0.8*len(preds)))
    # plotFNCount(preds, labels)
    # plotFNRate(preds, labels)
    # plotFPCount(preds, labels)
    # plotFPRate(preds, labels)

    # # Save to CSV file
    # df = pd.DataFrame([score])
    # df.to_csv("output.csv")

if __name__ == "__main__": main()