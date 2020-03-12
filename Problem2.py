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
	return unwarp_image


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

if __name__ == '__main__':
	img = np.copy(image_src)
	unwarp = unwarp(img)
	# cv2.imshow("output",unwarp)
	img1=np.copy(unwarp)
	undistort = undistort(img1)
	cv2.imshow("output1",undistort)
	cv2.waitKey(0)