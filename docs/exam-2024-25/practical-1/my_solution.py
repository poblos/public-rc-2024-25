import mujoco
import random
import numpy as np
import math
import cv2
from typing import Tuple

def is_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.03*cv2.arcLength(largest_contour,True)
    approx = cv2.approxPolyDP(largest_contour,epsilon,True)
    print(len(approx))
    if (len(approx) <= 4):
        return False
    return True

class Detector:
    def __init__(self) -> None:
        self.cages = 0
        self.best = 0
        self.best_result = False

    def detect(self, img) -> None:
        self.cages += 1
        #if self.cages % 400 == 0:
            #display_image(img)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1

        if np.sum(mask) > self.best:
            self.best_result = is_ball(mask)
            self.best = np.sum(mask)

        if self.cages % 40 == 0:
            print(f'{self.best} and {np.sum(mask)}')
            print(self.best_result)
    def result(self) -> str:
        if self.best == 0:
            return ""
        elif self.best_result == True:
            return "sphere"
        return "box"


class DetectorPos:
    def __init__(self) -> None:
        pass

    def detect(self, img) -> Tuple[float, float]:
        return 0, 0
