import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# STEP0. CALIBRATION BOARD BEFORE CAMERA CALIBRATION
images = glob.glob("camera_cal/calibration*.jpg")

# Chess board (9,6)
objpoints = []
imgpoints = []

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) 

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0]) # x, y
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def color_thresholding(img, threshold=(0,255), opt=("rgb")):
    # read using mpimg as R.G.B
    img_in = np.copy(img)
    
    if (opt == "rgb"):
        rgb = img_in
        r_channel = rgb[:,:,0]
        g_channel = rgb[:,:,1]
        b_channel = rgb[:,:,2]

        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= threshold[0]) & (r_channel <= threshold[1])]=1

        return r_binary
    
    elif (opt == "hls"):
        hls = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        print(s_channel)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= threshold[0]) & (s_channel <= threshold[1])]=1
        print(s_binary)

        return s_binary

    else:
        return img_in

def process_img(image):
    process_img.running_flag += 1
    
    img = np.copy(image)
    # step 1: undistortion
    undist = cal_undistort(img, objpoints, imgpoints)
    
    # step 2: Thresholding (Color, Gradient, Combination)
    color = color_thresholding(undist, threshold=(70, 255), opt="hls")

    return color
    
    # step 3: perspective(bird eye)

    # step 4: Search from Prior

    # step 5: measure Curvature

    # step 6: Inverse Warp

    # step 7: Visualization

process_img.running_flag=0
images = glob.glob("test_images/test*.jpg")

for fname in images:
    img = mpimg.imread(fname)
    res=process_img(img)
    plt.imshow(res,cmap='gray')
    plt.show()