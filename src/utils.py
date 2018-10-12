import numpy as np

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]  + 1) * (boxA[3]  + 1)
    boxBArea = (boxB[2]  + 1) * (boxB[3]  + 1)


    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def scale(boxA, boxB):
    width_A = abs(boxA[0] - boxA[2])
    height_A = abs(boxA[1] - boxA[3])

    width_B = abs(boxB[0] - boxB[2])
    height_B = abs(boxB[1] - boxB[3])

    scale = abs((width_A * height_A) / (width_B * height_B))

    return scale