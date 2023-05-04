import numpy as np
import csv
import os
import pandas as pd
from metrics import *
import random

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

def getHMLRates(label_dir, predictions_dir):
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

                # preds = load_predicions(os.path.join(predictions_dir, path, pred_file))
                preds = load_predictions2(os.path.join(predictions_dir, path, pred_file))

                labels = load_labels3(os.path.join(label_dir, l), 4)

                result = calculateRates(preds, labels, thresh = thresh)
                MCD.append(result['MCD'])
                FNR.append(result['FNR'])
                FPR.append(result['FPR'])

            all_metrics.append([path + f"-THRESH-{thresh}", sum(MCD) / len(MCD), sum(FNR) / len(FNR), sum(FPR) / len(FPR)])

    df = pd.DataFrame(all_metrics, columns=['Model', 'MCD', 'FNR', 'FPR'])
    df.to_csv('HML_rates.csv')

def getHMLMetrics(label_dir, predictions_dir):
    labels_files = [l for l in os.listdir(label_dir) if l.endswith('.avi.csv')]
    
    all_metrics = []

    for path in os.listdir(predictions_dir):
        files = os.listdir(os.path.join(predictions_dir, path))
        MAP = []
        FNC = []
        MCD = []

        for l in labels_files:
            pred_file = [f for f in files if f.split('.csv')[0] == l.split('.avi.csv')[0]][0]
            print(os.path.join(label_dir, l), os.path.join(predictions_dir, path, pred_file))

            # preds = load_predicions(os.path.join(predictions_dir, path, pred_file))
            preds = load_predictions2(os.path.join(predictions_dir, path, pred_file))

            labels = load_labels3(os.path.join(label_dir, l), 4)

            result = metrics(preds, labels, 0.5, 0.95)
            MAP.append(result['mAP'])
            FNC.append(result['FN_count'])
            MCD.append(result['mean_center_dist'])

        all_metrics.append([path, sum(MAP) / len(MAP), sum(FNC) / len(FNC), sum(MCD) / len(MCD)])

    df = pd.DataFrame(all_metrics, columns=['Model', 'mAP', 'FN_count', 'MCD'])
    df.to_csv('HML_rates.csv')

def getAirSimRates(prediction_path: str, label_path: str):
    models = [p for p in os.listdir(prediction_path) if p.startswith('airSim') and not p.endswith('onnx')]

    all_metrics = []

    models = sorted(models)

    for thresh in [10, 20, 30]:
        for model in [models[1]]:
            MCD = []
            FNR = []
            FPR = []

            print(model)
            cams = sorted([p for p in os.listdir(os.path.join(prediction_path, model)) if p.endswith('.csv')])

            for i, cam in enumerate(cams):
                print(cam)
                labels = load_labels(os.path.join(label_path, f'labelsCam{i+1}.csv'))
                preds = load_predicions(os.path.join(prediction_path, model, cam))

                result = calculateRates(preds, labels, thresh)
                MCD.append(result['MCD'])
                FNR.append(result['FNR'])
                FPR.append(result['FPR'])

            all_metrics.append([model + f"-THRESH-{thresh}", sum(MCD) / len(MCD), sum(FNR) / len(FNR), sum(FPR) / len(FPR)])

    df = pd.DataFrame(all_metrics, columns=['Model', 'MCD', 'FNR', 'FPR'])
    df.to_csv('AirSim_rates.csv')

def main():
    getHMLMetrics('./labels/labels3', './predictions/predictions4')
    # getAirSimRates('./AirSim/lab_12', './labels/labels4_lab_12')

    # plot lab sequences
    # plotAirSimData('./AirSim/lab_12', './labels/labels4_lab_12')

    # preds = load_predicions("./predictions/predictions3/best_640T0.2/Dron T02.55260362.20220117151609.csv") # One real drone
    # preds = load_predicions("./AirSim/lab_12/airSim_640T0.2/cam4i_s.csv")

    # colors = np.array([[255, 0, 0], # Red
    #                [0, 255, 0], # Green
    #                [0, 0, 255], # Blue
    #                [255, 255, 0], # Yellow
    #                [255, 0, 255], # Magenta
    #                [0, 255, 255], # Cyan
    #                [127, 127, 127], # Brown
    #                [255, 127, 32]]) # Olive
    # curr_col = 0

    # image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    
    # # Create paths from first frame
    # paths = []
    # for p in [p for p in preds if p[0] == 1]:
    #     y = int(p[2] + (p[4] - p[2]) / 2)
    #     x = int(p[1] + (p[3] - p[1]) / 2)
    #     paths.append({'color': colors[curr_col], 'x': x, 'y': y})
    #     curr_col += 1

    # for j in range(1, 3):
    #     pred = [p for p in preds if p[0] == j]
    #     if len(pred) == 0: continue

    #     for path in paths:
    #         closest_point = None
    #         min_dist = 1e+6
    #         for i, point in enumerate(pred):
    #             y = int(point[2] + (point[4] - point[2]) / 2)
    #             x = int(point[1] + (point[3] - point[1]) / 2)
    #             dist = math.sqrt((path['x'] - x) ** 2 + (path['y'] - y) ** 2)
    #             if dist < min_dist:
    #                 closest_point = i
    #                 min_dist = dist
    #         print(closest_point)
    #         x = int(pred[closest_point][1] + (pred[closest_point][3] - pred[closest_point][1]) / 2)
    #         y = int(pred[closest_point][2] + (pred[closest_point][4] - pred[closest_point][2]) / 2)
    #         path['x'] = x
    #         path['y'] = y
    #         image[y:y+5, x:x+5] = path['color']

    # cv2.imwrite("output.png", image)
    # visualizeDataset(preds, labels, video='./predictions3/best_640T0.2/Dron T02.55260362.20220117151609.mp4')

    # score = metrics(preds, labels, mAP_start=0.5, mAP_stop=0.55, mAP_step=0.05, main_iou_thresh=0.5, dist_thresh=20, plot=False)
    # print(score)

if __name__ == "__main__": main()