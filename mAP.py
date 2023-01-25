import cv2
import numpy as np
import csv
import matplotlib
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid 
import pandas as pd
import tensorflow as tf



class bbox():
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __init__(self, arr):
        self.x = arr[0]
        self.y = arr[1]
        self.w = arr[2]/2
        self.h = arr[3]/2
    def intersection_over_union(self, boxB):
        l1 = max(self.x - self.w, boxB.x - boxB.w)
        t1 = max(self.y - self.h, boxB.y - boxB.h)
        l2 = min(self.x + self.w, boxB.x + boxB.w)
        t2 = min(self.y + self.h, boxB.y + boxB.h)
        inter = min(0, (l1-l2)) * min(0, (t1-t2))
        union = 4*self.w*self.h + 4*boxB.w*boxB.h - inter
        iou = inter/union
        return iou


DRAW = True
LAB = 5
# DRAW = False

def intersection_over_union(boxA, boxB):
    l1 = max(boxA.x - boxA.w, boxB.x - boxB.w)
    t1 = max(boxA.y - boxA.h, boxB.y - boxB.h)
    l2 = min(boxA.x + boxA.w, boxB.x + boxB.w)
    t2 = min(boxA.y + boxA.h, boxB.y + boxB.h)
    inter = min(0, (l1-l2)) * min(0, (t1-t2))
    union = 4*boxA.w*boxA.h + 4*boxB.w*boxB.h - inter
    iou = inter/union
    return iou



def load_predicted_data(df):
    drones_by_frame = []
    for i in range(1,np.max(df['frame_num'])+1):#dla wszystkich klatek
        rows = df.loc[df['frame_num'] == i].get(['x_c','y_c','w','h'])#wybierz wszstkie wiersze o itym numerze klatki
        drones_by_frame.append(np.array(rows))
    return drones_by_frame

def load_true_data(df:pd.DataFrame()):
    frame_num = df.get(0)
    df = df.drop(columns =0)
    drones_by_frame = []
    for i in range(0,np.max(frame_num)):
        drones_by_frame.append(np.array(df.loc[i].dropna()).reshape(-1,4))
    return drones_by_frame    

def to_bbox(drones):
    drones_by_frame = []
    for frame in drones:
        drones_by_frame.append([bbox(drone) for drone in frame])
    return drones_by_frame


def pair_bbox(real_drones,detected_drones):
    pair_of_bbox = []
    for i,real_drone_frame in enumerate(real_drones):
        detected_drone_frame = detected_drones[i].copy()
        # przyporządkowujemy każdemu dronowi detekcję z którą ma największe IoU 
        for drone in real_drone_frame:
            iou = []
            for i,drone_p in enumerate(detected_drone_frame):
                iou.append(intersection_over_union(drone, drone_p))

            if len(iou)>0:
                pair_of_bbox.append((drone, detected_drone_frame[np.argmax(iou, 0)]))
                detected_drone_frame.pop(np.argmax(iou, 0))
    return pair_of_bbox




def main():
    thresholds = np.arange(start=0.0, stop=1.1, step=0.1)
    for camera in range(1, 9):  # Dla każdej z 8 kamer
        precisions = []
        recalls = []

        true_positions =  pd.read_csv(".\\Dataset\\{}\\Data\\connected_{}.csv".format(LAB,camera), delimiter=";", header=None)
        detected_positions = pd.read_csv(".\\Dataset\\{}\\Exps\\{}\\coordinates.csv".format(LAB,camera),delimiter=',')
        
        detected_bbox = load_predicted_data(detected_positions)
        true_bbox = load_true_data(true_positions)

        
        real_drones = to_bbox(true_bbox)
        detected_drones = to_bbox(detected_bbox)
        pair_of_bbox = pair_bbox(real_drones,detected_drones)
        ious = [intersection_over_union(d,t) for d,t in pair_of_bbox]

        n_detected = np.sum([len(d) for d in detected_drones])
        n_real = np.sum([len(r) for r in real_drones])
        pr = []
        # Oblicznie ilości dobrze przyporządkowanych dronów przy założonym progu
        for t in thresholds:
            tp = sum(i > t for i in ious)

            # obliczenie precision i recall
            if not n_detected == 0:
                precision = tp/n_detected
            else:
                precision = 1
            if not n_real == 0:
                recall = tp/n_real
            else:
                recall = 0
            pr.append((precision,recall))
        pr.sort(key=lambda x: x[1])
        precisions,recalls = zip(*pr)
        reinterpreted_precision = [np.max(precisions[i:]) for i in range(len(precisions))]
        if np.max(recalls) - np.min(recalls):
            Ap = np.average(precisions,weights=recalls)
            Ap = trapezoid(reinterpreted_precision,recalls)/(np.max(recalls) - np.min(recalls))
            # Ap = np.average(precisions)
            print("Camera: {}, Avarage precision: {}".format(camera,Ap))

        
        plt.plot(recalls, reinterpreted_precision, linewidth=4, color="red", zorder=0)
        plt.xlabel("Recall", fontsize=12, fontweight='bold')
        plt.ylabel("Precision", fontsize=12, fontweight='bold')
        plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
        plt.show()
        # print("camera: ", camera, " mAP: ", np.sum(mAP)/len(mAP))
        # plt.plot(tresh_to_show, mAP)
        # plt.ylabel("mAP", fontsize=12, fontweight='bold')
        # plt.xlabel("thresholds", fontsize=12, fontweight='bold')
        # plt.show()


if __name__ == '__main__':
    main()
