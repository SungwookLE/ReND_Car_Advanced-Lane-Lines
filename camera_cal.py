import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Chess board (9,6)
objpoints = []
imgpoints = []

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) 

# STEP0. CALIBRATION BOARD BEFORE CAMERA CALIBRATION
images = glob.glob("camera_cal/calibration*.jpg")

def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0]) # x, y
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

idx=0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        idx+=1
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        undist_img = cal_undistort(img, objpoints, imgpoints)
        cv2.imshow('img'+str(idx), img)
        cv2.imshow('undist_img'+str(idx), undist_img)

cv2.waitKey(0)


