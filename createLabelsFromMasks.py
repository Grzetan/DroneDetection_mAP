import csv
import numpy as np
import cv2
import os

def createMask(path: str, show: bool = True) -> tuple:
    """! Calculates bounding box for given mask.
    @param path Path to binary mask
    @param show Decides if calcualted bbox should be shown
    @return Returns tuple of (valid, bbox). Valid means that bbox was indeed detected and bbox is a 5-element array [frame_id, x1, y1, x2, y2]
    """
    img = cv2.imread(path, 0)
    # img = cv2.resize(img, (640, 640))
    points = np.transpose(np.nonzero(img == 255))
    x,y,w,h = cv2.boundingRect(points)

    # Get frame idx
    frame = int(path.split('_')[-1].split('.')[0])

    if show:
        img = cv2.rectangle(img,(y,x),(y+h,x+w),(255,255,255),1)
        cv2.imshow(path, img)
        cv2.waitKey(0)

    if x == 0 and y == 0 and w == 0 and h == 0:
        return False, []

    return True, np.array([frame,y,x,y+h,x+w])

def main():
    output_dir = './labels4_lab_15'
    input_dir = './AirSim/lab_15'

    # Loop through every camera
    for cam in range(1, 9):
        print("Cam", cam)

        # Create output file for every cam
        f = open(os.path.join(output_dir, f'labelsCam{cam}.csv'), 'w')
        writer = csv.writer(f)

        # Loop through every drone and create output file
        drones = sorted([p for p in os.listdir(input_dir) if p.startswith('mask')])
        for drone in drones:
            # Get every frame from current cam
            masks = [m for m in os.listdir(os.path.join(input_dir, drone)) if m.startswith(f'cam{cam}')]
            # For every mask create bbox and save it to current cam output file
            for mask in masks:
                valid, bbox = createMask(os.path.join(input_dir, drone, mask), show=False)
                if valid:
                    writer.writerow(bbox)

        f.close()


    # path = './AirSim/lab_12/mask1/cam1b_s_450.jpeg'
    # createMask(path)

if __name__ == '__main__':
    main()