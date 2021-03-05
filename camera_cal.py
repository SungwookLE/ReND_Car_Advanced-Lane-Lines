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
        cv2.imshow('img'+str(idx), img)
        cv2.waitKey(100)


        
def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0]) # x, y
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def process_img(image):
    process_img.running_flag += 1
    
    img = np.copy(image)

    # step 1: undistortion
    undist = cal_undistort(img, objpoints, imgpoints)

    # step 2: Thresholding (Color, Gradient, Combination)

    # step 3: perspective(bird eye)

    # step 4: Search from Prior

    # step 5: measure Curvature

    # step 6: Inverse Warp

    # step 7: Visualization

    



