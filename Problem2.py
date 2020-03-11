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
	return unwarp_image, M, MinV


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
	hls = cv2.cvtcolor(image,cv2.COLOR_RGB2HLS)
	hls_l = hls[:,:,1]
	hls_l = hls_l*(255/np.max(hls_l))
	binary_out = np.zerosl_like(hls_l)
	binary_out[hls_l>threshold[0] & (hls <= threshold[1])] = 1
	return binary_out

# Thresholding using lab 
def lab(image, threshold = (190,255)):
	lab = cv2.cvtcolor(image,cv2.COLOR_RGB2Lab)
	lab_b = lab[:,:,2]
	if np.max(lab_b)>175:
	lab_b = lab_b*(255/np.max(lab_b))
	binary_out  = np.zerosl_like(lab_b)
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
def kalman():
	global kalman
	kalman_filter = cv2.kalmanFilter(4,2)
	kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

    kalman.transitionMatrix = np.float32([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]])

    kalman.processNoiseCov = np.float32([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]]) * 0.03

    Kalman_measurement = np.float32((2,1))
    kalman_prediction = np.zeros((2,1), np.float32)

    

# Kalman Prediction

def Predict(points):
	kalman.correct(points)
    kalman_prediction = kalman.predict()
    return kalman_prediction


if __name__ == '__main__':
	# img = np.copy(image_src)
	# unwarp = unwarp(img)
	# # cv2.imshow("output",unwarp)
	# img1=np.copy(unwarp)
	# undistort = undistort(img1)
	# cv2.imshow("output1",undistort)
	# cv2.waitKey(0)