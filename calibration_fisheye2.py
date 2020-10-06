#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import math


CHECKERBOARD = (5, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objpoints = []
imgpoints = [] 
_img_shape = None

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

images = glob.glob('images_gathered/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    #img = cv2.resize(img, (400,300))
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

h,w = img.shape[:2]

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM = " + str(_img_shape[::-1]))
print("K (intrinsic_matrix):")
print(K)
print("D (distortion coeffs):")
print(D)

DIM=_img_shape[::-1]
balance=0.98
dim2=None
dim3=None

img = cv2.imread('image_test/noise.jpg')
cv2.imshow('img before undistorting',img)
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
print('scaled_K', scaled_K)
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

with open('scaled_K.npy', 'wb') as f:
    np.save(f, np.array(scaled_K))

with open('new_K.npy', 'wb') as f:
    np.save(f, np.array(new_K))

with open('D.npy', 'wb') as f:
    np.save(f, np.array(D))
    

map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow('undistorted_img',undistorted_img)
cv2.imwrite('undistorted_img.jpg', undistorted_img)

###### LETS CHANGE undistorted_img
# objpoints2 = []
# imgpoints2 = [] 

# objp2 = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp2[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# prev_img_shape = None

# img_und = undistorted_img
# gray2 = cv2.cvtColor(img_und,cv2.COLOR_BGR2GRAY)

# ret2, corners_und = cv2.findChessboardCorners(gray2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

# objpoints2.append(objp2)
# corners2_und = cv2.cornerSubPix(gray, corners_und, (11,11),(-1,-1), criteria)
# imgpoints2.append(corners2_und)
# img_und = cv2.drawChessboardCorners(img_und, CHECKERBOARD, corners2_und, ret2)
# print(imgpoints2)
# # imgpts = np.asarray(imgpoints2)
# # imgpts = np.reshape(imgpts, 2)
# # print(imgpts)
# # print(imgpts[0])
# #print(imgpts[0][1])
# cv2.imshow('img_und',img_und)
cv2.waitKey(0)
