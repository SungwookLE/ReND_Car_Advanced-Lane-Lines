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
        r_channel = cv2.equalizeHist(r_channel)
        r_binary[(r_channel >= threshold[0]) & (r_channel <= threshold[1])]=1

        return r_binary
    
    elif (opt == "hls"):
        hls = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        s_binary = np.zeros_like(s_channel)
        s_channel = cv2.equalizeHist(s_channel)
        s_binary[(s_channel >= threshold[0]) & (s_channel <= threshold[1])]=1

        return s_binary

    else:
        return img_in

def gradient_thresholding(img, threshold=(0,255), opt=("comb")):
    # read using mpimg as R.G.B
    img_in = np.copy(img)
    gray= cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

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

def perspective_img(image, region_rect, mode=("normal")):
    img_in = np.copy(image)

    x_len = img_in.shape[1]
    y_len = img_in.shape[0]

    if (mode == "normal"):
        src_pts = np.array(region_rect)
        margin=50
        warp_rect = np.array([[margin, margin] ,[x_len-margin, margin], [x_len-margin, y_len-margin], [margin, y_len-margin]], np.float32)
        out_pts = np.array(warp_rect)
    
    else: #inverse
        margin=50
        warp_rect = np.array([[margin, margin] ,[x_len-margin, margin], [x_len-margin, y_len-margin], [margin, y_len-margin]], np.float32)
        src_pts = np.array(warp_rect)
        out_pts = np.array(region_rect)

    M = cv2.getPerspectiveTransform(src_pts, out_pts)
    warp = cv2.warpPerspective(img_in, M, (x_len, y_len))
    
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
        left_fit_coef = None
        right_fit_coef = None
    
    return left_fitx, right_fitx, ploty, left_fit_coef, right_fit_coef

def search_around_poly(binary_warp, init_tune):
    binary_warped = np.copy(binary_warp)

    margin = 100

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

    left_fitx, right_fitx, ploty,  left_fit_coef, right_fit_coef = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)
    left_right_coeff = np.array([left_fit_coef, right_fit_coef])

    ## VISUALIZATION FOR TESTING
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='blue')

    return binary_warped, left_fitx, right_fitx, ploty, left_right_coeff

def measure_curvature(image, left_fitx, right_fitx, ploty, ratio=(1,1)):

    # Image x size
    img_x_size = image.shape[1] 

    left_fit_cr = np.polyfit(ploty * ratio[1], left_fitx * ratio[0], 2)
    right_fit_cr = np.polyfit(ploty * ratio[1], right_fitx * ratio[0], 2)

    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = (
        1 + (2 * left_fit_cr[0] * y_eval * ratio[1] + left_fit_cr[1]) ** 2
    ) ** 1.5 / np.absolute(2 * left_fit_cr[0])
    right_curverad = (
        1 + (2 * right_fit_cr[0] * y_eval * ratio[1] + right_fit_cr[1]) ** 2
    ) ** 1.5 / np.absolute(2 * right_fit_cr[0])

    mean_curverad = np.mean([left_curverad, right_curverad])

    left_x = (
        left_fit_cr[0] * (y_eval * ratio[1]) **2 
        + left_fit_cr[1] * (y_eval * ratio[1])
        + left_fit_cr[2]
    )
    left_of_center = (img_x_size / 2) * ratio[0] - left_x


    return left_curverad, right_curverad, mean_curverad, left_of_center

def process_img(image):
    process_img.running_flag += 1
    
    img = np.copy(image)

    # step 1: undistortion
    undist = cal_undistort(img, objpoints, imgpoints)

    # step 2: Thresholding (Color, Gradient, Combination)
    color = color_thresholding(undist, threshold=(55, 255), opt="hls")
    gradient_comb = gradient_thresholding(undist,threshold=(60, 200), opt="comb" )
    gradient_dir = gradient_thresholding(undist,threshold=(0*np.pi/180,50*np.pi/180), opt="dir" )

    thd_img = np.zeros_like(gradient_comb)
    thd_img[
        (color ==1) & ((gradient_comb==1) | (gradient_dir==1)) 
    ]=1

    
    # step 3: perspective(bird eye)
    region_rect= np.array([[490, 535], [835, 535], [1080, 650], [265, 650]], np.float32)

    ## DEBUG: ROI CHECK
    #rect_samp = np.array([[490, 515], [835, 515], [1080, 650], [265, 650]], np.int)
    #cv2.polylines(thd_img, [rect_samp], True, (255,255,255), 10)
    #return thd_img

    warp_img = perspective_img(thd_img, region_rect, mode="normal")

    # step 4: Search from Prior
    init_left_fit = np.array([8.22279110e-05, -8.01574626e-02, 1.80496286e02])
    init_right_fit = np.array([9.49537809e-05, -9.58782039e-02, 1.18196061e03])
    if ( process_img.running_flag < 2 or (process_img.left_right_coeff[0] is None) or (process_img.left_right_coeff[1] is None) ):
        init_tune = np.array([init_left_fit, init_right_fit])
    else:
        if ( np.sum(np.abs(init_left_fit - process_img.left_right_coeff[0])) < 500   ) and ( np.sum(np.abs(init_right_fit - process_img.left_right_coeff[1])) < 600 ):
            init_tune = process_img.left_right_coeff
        else:
            init_tune = np.array([init_left_fit, init_right_fit])

    polyfit_img, left_fitx, right_fitx, ploty, process_img.left_right_coeff = search_around_poly(warp_img, init_tune)

    # step 5: measure Curvature.
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 1000  # meters per pixel in x dimension
    ratio = [xm_per_pix, ym_per_pix]
    left_curverad, right_curverad, mean_curverad, left_of_center = measure_curvature(polyfit_img, left_fitx, right_fitx, ploty, ratio=ratio)
    #print("Mean_CurveRad: ", mean_curverad, "Left_of_Center: ", left_of_center)

    str_curv = (
        "Radius of Curvature = %6d" % mean_curverad
        + "(m)     Vehicle is %.2f" % left_of_center
        + "m left of center"
    )

    # step 6: Inverse Warp
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(polyfit_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the Lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    draw_line_left = np.int_(np.column_stack([left_fitx, ploty]))
    draw_line_right = np.int_(np.column_stack([right_fitx, ploty]))
    cv2.polylines(color_warp, [draw_line_left], False, (255,0,0), 25)
    cv2.polylines(color_warp, [draw_line_right], False, (255,0,0), 25)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Lane_warp_back = perspective_img(color_warp, region_rect, mode="inverse")

    # step 7: Visualization
    # Combine the result with the original image
    result_img = cv2.addWeighted(undist, 1, Lane_warp_back, 0.5, 0)
    
    cv2.putText(
        result_img,
        str_curv,
        (10, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0,0,255),
        thickness=2,
    )
    
    return result_img

"""
images = glob.glob("test_images/test*.jpg")
for fname in images:
    process_img.running_flag=0
    img = mpimg.imread(fname)
    res=process_img(img)
    plt.imshow(res,cmap='gray')
    plt.show()
"""

from moviepy.editor import VideoFileClip
from IPython.display import HTML

videos = glob.glob("*.mp4")
for fname in videos:
    process_img.running_flag = 0
    process_img.left_right_coeff = np.array([None, None])

    src_string = fname
    src_video = VideoFileClip(src_string)
    out_video = "output_videos/out_" + src_string

    img_clip = src_video.fl_image(process_img)
    img_clip.write_videofile(out_video, audio=False)

