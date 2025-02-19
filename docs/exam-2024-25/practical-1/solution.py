import mujoco
import random
import numpy as np
import math
import cv2
from typing import Tuple

def is_ball(mask):
    sums = np.sum(mask, axis=0)
    sums = sums[sums > 10]
    return np.quantile(sums, 0.15) / sums.max() < 0.72

def red_mask(img_hsv):
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    return mask0 + mask1

class Detector:
    def __init__(self) -> None:
        self.detected = ''
        self.best = 10000

    def detect(self, img) -> None:
        img = img[:, 200 : 640 - 200, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = red_mask(img_hsv)

        if np.sum(mask) > self.best:
            cv2.imwrite("f.png", img)
            self.detected = 'sphere' if is_ball(mask) else 'box'
            self.best = np.sum(mask)

    def result(self) -> str:
        return self.detected


class DetectorPos:
    def __init__(self) -> None:
        pass

    def detect(self, img) -> Tuple[float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x_mask = np.sum(np.sum(img_hsv, axis=2), axis=0) > 0
        y_mask = np.sum(np.sum(img_hsv, axis=2), axis=1) > 0
        (y_size, x_size) = (len(img_hsv), len(img_hsv[0]))
        mean_x_pos = np.argwhere(x_mask>0).reshape(-1).mean() - (x_size / 2)
        mean_y_pos = np.argwhere(y_mask>0).reshape(-1).mean() - (y_size / 2)

        return (mean_y_pos + 0.5) / 230, (-mean_x_pos - 23.5) / 230
