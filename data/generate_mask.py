import numpy as np
import cv2
import random

def init_mask(input_size):
    msk = np.zeros((input_size, input_size))

    border_limit_inf = input_size // 10
    border_limit_sup = input_size - border_limit_inf
    x = random.randrange(border_limit_inf, border_limit_sup)
    y = random.randrange(border_limit_inf, border_limit_sup)
    center_pt = (x,y)
    radius = random.randrange(0,10)
    color = (255, 255, 255)
    thickness = 1
    # Apply circle
    msk = cv2.circle(msk, center_pt, radius, color, thickness)
    # Apply random speeds
    for i in range(4):
        speeds = random.sample(range(-30, 30), 2)
        radius = random.randrange(20, 30)
        msk = cv2.circle(msk, center=(x + speeds[0], y + speeds[1]), radius=radius, color=color)
        msk = cv2.line(msk, center_pt, (x + speeds[0], y + speeds[1]), color=color)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    msk = cv2.dilate(msk, kernel_ellipse, iterations=4)

    return msk