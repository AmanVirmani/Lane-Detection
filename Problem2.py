import numpy as np
import cv2
import matplotlib.pyplot as plt

# Experimenting with one frame of the video
image_src  = cv2.imread('Figure_1.png')


# Defining fuction for unwarping
def unwarp(image): 
	height = image.shape[0]
	width = image.shape[1]
	point_src = np.float32([(0.45*width,0.65*height),
		(0.60*width,0.64*height),
		(0.10*width,height),
		(0.97*width,height)])
	point_dst = np.float32([(456,0),
		(width-456,0),
		(456,height),
		(width-456,height)])

	# Computing homography
	H= cv2.getPerspectiveTransform(point_src,point_dst)
	MinV = cv2.getPerspectiveTransform(point_dst,point_src)

	# To get bird's eye view
	unwarp_image = cv2.warpPerspective(image,H,(width,height), flags= cv2.INTER_LINEAR)
	return unwarp_image, H, MinV


# # Output: after Unwarping the source image
# img = np.copy(image_src)
# unwarp = unwarp(img)
# cv2.imshow("output",unwarp)
# cv2.waitKey(0)

# Defining Function for Undistortion of the Image
def undistort(input_image):
	camera_matrix = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
	dist_coeff = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])
	undistort_image = cv2.undistort(unwarp,camera_matrix, dist_coeff, None, camera_matrix)
	return undistort_image

# Thresholding using HLS color space
def hls(image, threshold = (220,255)):
	hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	hls_l = hls[:,:,1]
	hls_l = hls_l*(255/np.max(hls_l))
	binary_out = np.zeros_like(hls_l)
	binary_out[hls_l>threshold[0] & (hls <= threshold[1])] = 1
	return binary_out

# Thresholding using lab 
def lab(image, threshold = (190,255)):
	lab = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
	lab_b = lab[:,:,2]
	if np.max(lab_b)>175:

		lab_b = lab_b*(255/np.max(lab_b))
		binary_out  = np.zeros_like(lab_b)
		binary_out[((lab_b>threshold[0])&(lab_b <= threshold[1]))]
		return binary_out

# Image Pipeline
def pipeline(image):
	image_unwarp,H,MinV = unwarp(image)
	imagel_threshold = hls(image_unwarp)
	imageb_threshold = lab(image_unwarp)
	combined_image = np.zeros_like(imageb_threshold)
	combined_image[(imagel_threshold == 1) | (imageb_threshold == 1)]= 1
	return combined_image,MinV

# Kalman filter
def Kalman():
    global kalman
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)

    kalman.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.03

    measurement = np.array((2,1), np.float32)
    prediction = np.zeros((2,1), np.float32)
    
# kalman Prediction
def prediction(points):
    kalman.correct(points)
    kalman_prediction = kalman.predict()
    return kalman_prediction

# sliding Window
def sliding_window(image):
	global kalman
	hist = np.sum(imgae[image.shape[0]//2:,:],axis = 0)
	mid_point = np.int(hist.shape[0]//2)
	quarter_point = np.int(midpoint//2)
	left_x_base = np.argmax(hist[quarter_point:mid_point]) + quarter_point
	right_x_base = np.argmax(hist[mid_point:(mid_point+quarter_point)]) + mid_point
	Windows = 10
	window_height = np.int(image.shape[0]/windows)
	non_zero = image.non_zero()
	non_zero_x = np.array(non_zero[1])
	non_zero_y = np.array(non_zero[0])
	right_x_current = kalman_prediction[1]
	left_x_current = kalman_prediction[0]
	window_margin = 80
	min_pixels = 40
	left_lane_indices = [] 
	right_lane_indices = []
	rectangle_data = []

	for window in range(windows):


		# Identified window boundaries in x and y (and right and left)
		win_y_low = imgage.shape[0] - (window+1)*window_height
		win_y_high = imgage.shape[0] - window*window_height
		win_xleft_low = left_x_current - window_margin
		win_xleft_high = left_x_current + window_margin
		win_xright_low = right_x_current - window_margin
		win_xright_high = right_x_current + window_margin
		rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

		# Identified the nonzero pixels in x and y within the window
		good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]


		left_lane_indices.append(good_left_indices)
		right_lane_indices.append(good_right_indices)


		if len(good_left_indices) > min_pixels:
		    left_x_current = np.int(np.mean(nonzerox[good_left_indices]))
		if len(good_right_indices) > minpix:        
		    right_x_current = np.int(np.mean(nonzerox[good_right_indices]))

	left_lane_indices = np.concatenate(left_lane_indices)
	right_lane_indices = np.concatenate(right_lane_indices)

	
	leftx = nonzerox[left_lane_indices]
	lefty = nonzeroy[left_lane_indices] 
	rightx = nonzerox[right_lane_indices]
	righty = nonzeroy[right_lane_indices] 

	left_fit, right_fit = (None, None)

	if len(leftx) != 0:
	    left_fit = np.polyfit(lefty, leftx, 2)
	if len(rightx) != 0:
	    right_fit = np.polyfit(righty, rightx, 2)

	visualization_data = (rectangle_data, histogram)
	return left_fit, right_fit, left_lane_indices, right_lane_indices, visualization_data ,left_x_base ,right_x_base 

# Visualization
exampleImg = cv2.imread('Test.png')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg_bin, MinV = pipeline(exampleImg)
    
left_fit, right_fit, left_lane_indices, right_lane_indices, visualization_data = sliding_window(exampleImg_bin)

height = exampleImg.shape[0]
left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
#print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

rectangles = visualization_data[0]
histogram = visualization_data[1]

# Create an output image to draw on and  visualize the result
out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin))*255)
# Generate x and y values for plotting
ploty = np.linspace(0, exampleImg_bin.shape[0]-1, exampleImg_bin.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
for rect in rectangles:
# Draw the windows on the visualization image
    cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
    cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
# Identify the x and y positions of all nonzero pixels in the image
nonzero = exampleImg_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [100, 200, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
# if __name__ == '__main__':
	# img = np.copy(image_src)
	# unwarp = unwarp(img)
	# # cv2.imshow("output",unwarp)
	# img1=np.copy(unwarp)
	# undistort = undistort(img1)
	# cv2.imshow("output1",undistort)
	# cv2.waitKey(0)