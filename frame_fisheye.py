#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import math

with open('scaled_K.npy', 'rb') as f:
    scaled_K = np.load(f)

with open('new_K.npy', 'rb') as f:
    new_K = np.load(f)

with open('D.npy', 'rb') as f:
    D = np.load(f)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    dim3 = frame.shape[:2][::-1]
    cv2.imshow('fisheye image', frame)  

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('undistorted_img',undistorted_img)

    undistorted2_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('undistorted2_img',undistorted2_img)

    if cv2.waitKey(33) == ord('q'):
        print('exit')
        break

cv2.waitKey(0)
