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

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= threshold[0]) & (s_channel <= threshold[1])]=1

        return s_binary

    else:
        return img_in

def gradient_thresholding(img, threshold=(0,255), opt=("comb")):
    # read using mpimg as R.G.B
    img_in = np.copy(img)
    gray= cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    img_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3)
    img_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3)

    abs_sobelx = np.absolute(img_sobel_x)
    abs_sobely = np.absolute(img_sobel_y)

    scaled_sobelx = np.uint8(
        255*abs_sobelx / np.max(abs_sobelx)
    )
    scaled_sobely = np.uint8(
        255*abs_sobely / np.max(abs_sobely)
    )

    img_sobel_xy = np.sqrt(img_sobel_x**2 + img_sobel_y**2)
    scaled_sobelxy = np.uint8(
        255*img_sobel_xy / np.max(img_sobel_xy)
    ) 

    direction = np.arctan2(abs_sobelx, abs_sobely)

    if (opt == "comb"):
        
        binary_comb = np.zeros_like(scaled_sobelxy)
        binary_comb[
            (scaled_sobelxy >= threshold[0]) & (scaled_sobelxy <= threshold[1])
        ]=1

        return binary_comb
    
    elif (opt == "x"):
        
        binary_x = np.zeros_like(scaled_sobelx)
        binary_x[
            (scaled_sobelx >= threshold[0]) & (scaled_sobelx <= threshold[1])
        ]=1

        return binary_x

    elif (opt == "y"):
        
        binary_y = np.zeros_like(scaled_sobely)
        binary_y[
            (scaled_sobely >= threshold[0]) & (scaled_sobely <= threshold[1])
        ]=1

        return binary_y

    elif (opt =="dir"):

        binary_dir = np.zeros_like(direction)
        binary_dir[
            (direction >= threshold[0]) & (direction <= threshold[1])
        ]=1

        return binary_dir

    else:
        return img_in
def perspective_img(image, region_rect):
    img_in = np.copy(image)

    x_len = img_in.shape[1]
    y_len = img_in.shape[0]
    
    src_pts = np.array(region_rect)

    margin=50
    warp_rect = np.array([[margin, margin] ,[x_len-margin, margin], [x_len-margin, y_len-margin], [margin, y_len-margin]], np.float32)
    out_pts = np.array(warp_rect)

    M = cv2.getPerspectiveTransform(src_pts, out_pts)
    warp = cv2.warpPerspective(img_in,M, (x_len, y_len) )
    
    return warp

def fit_polynomial(img_shape, leftx, lefty, rightx, righty):
    
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    try: #2nd order linear model was fitted using np.polyfit
        left_fit_coef = np.polyfit(lefty,leftx,2)
        right_fit_coef = np.polyfit(righty, rightx, 2)

        left_fitx= left_fit_coef[0]*ploty**2 + left_fit_coef[1]*ploty + left_fit_coef[2]
        right_fitx = right_fit_coef[0]*ploty**2 + right_fit_coef[1]*ploty + right_fit_coef[2]

    except TypeError:
        left_fitx = ploty
        right_fitx = ploty
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warp, init_tune):
    binary_warped = np.copy(binary_warp)

    margin = 150

    nonzero = binary_warped.nonzero() # nonzero index return!
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    left_lane_inds = (
        nonzerox
        > (
            init_tune[0][0] * (nonzeroy) ** 2
            + init_tune[0][1] * nonzeroy
            + init_tune[0][2]
            - margin
        )
    ) & (
        nonzerox
        < (
            init_tune[0][0] * (nonzeroy) ** 2
            + init_tune[0][1] * nonzeroy
            + init_tune[0][2]
            + margin
        )
    )

    right_lane_inds = (
        nonzerox
        > (
            init_tune[1][0] * (nonzeroy) ** 2
            + init_tune[1][1] * nonzeroy
            + init_tune[1][2]
            - margin
        )
    ) & (
        nonzerox
        < (
            init_tune[1][0] * (nonzeroy) ** 2
            + init_tune[1][1] * nonzeroy
            + init_tune[1][2]
            + margin
        )
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)

    #VISUALIZATION
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='blue')

    return binary_warped, left_fitx, right_fitx, ploty

def process_img(image):
    process_img.running_flag += 1
    
    img = np.copy(image)
    # step 1: undistortion
    undist = cal_undistort(img, objpoints, imgpoints)
    
    # step 2: Thresholding (Color, Gradient, Combination)
    color = color_thresholding(undist, threshold=(70, 250), opt="hls")
    gradient_comb = gradient_thresholding(undist,threshold=(20, 100), opt="comb" )
    gradient_dir = gradient_thresholding(undist,threshold=(0*np.pi/180,60*np.pi/180), opt="dir" )

    thd_img = np.zeros_like(gradient_comb)
    thd_img[
        (color ==1) & ((gradient_comb==1) & (gradient_dir==1)) 
    ]=1
    
    # step 3: perspective(bird eye)
    region_rect= np.array([[490, 515], [835, 515], [1080, 650], [265, 650]], np.float32)
    warp_img = perspective_img(thd_img, region_rect)


    # step 4: Search from Prior
    left_fit = np.array([8.22279110e-05, -8.01574626e-02, 1.80496286e02])
    right_fit = np.array([9.49537809e-05, -9.58782039e-02, 1.18196061e03])
    init_tune = np.array([left_fit, right_fit])

    polyfit_img, left_fitx, right_fitx, ploty = search_around_poly(warp_img, init_tune)

    return polyfit_img

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