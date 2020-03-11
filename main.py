import cv2
import argparse
import yaml
import numpy as np
import glob
import matplotlib.pyplot as plt

def removeDistortion(img,mtx,dist):
	h,  w = img.shape[:2]
	#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	newcameramtx= mtx
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
	return dst

def main(args):
	src_pts = np.float32([
		(0,379),
		(451,275),
		(726,275),
		(850,379)
	])

	dst_pts = np.float32([
		(0,300),
		(0,0),
		(300,0),
		(300,300)
	])

	K,D = readCamParams(args['cam_params'])
	images = glob.glob(args['image_path']+'/*')
	images.sort()
	for img_path in images:
		img = cv2.imread(img_path)
		cv2.imshow('before',img)
		cv2.waitKey(0);#cv2.destroyAllWindows()
		img = removeDistortion(img,K,D)
		cv2.imshow('after',img)
		cv2.waitKey(0)#;cv2.destroyAllWindows()
		dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		cv2.imshow('after denoising',dst)
		cv2.waitKey(0);#cv2.destroyAllWindows()
		edge = auto_canny(dst,0.99)
		roi = edge.copy();roi[:int(edge.shape[0]/2):,:] = 0 #region_of_interest(edge)
		cv2.imshow('roi', roi)
		cv2.imshow('edges', edge)
		cv2.waitKey(0); cv2.destroyAllWindows()

		M = cv2.getPerspectiveTransform(src_pts,dst_pts)
		invM = cv2.getPerspectiveTransform(dst_pts,src_pts)
		img = cv2.warpPerspective(edge,M,(300,300))
		cv2.imshow("bird's eye",img)
		cv2.waitKey(0)#;cv2.destroyAllWindows()

		break


if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-cam", "--cam_params", required=False, help="Camera Parameters file for the input data", default='../data/data_1/camera_params.yaml', type=str)
	ap.add_argument("-i", "--image_path", required=False, help="Path for input images", default='../data/data_1/data/', type=str)
	args = vars(ap.parse_args())

	main(args)

## TODO: remove all lines with slope less than 30 deg
## TODO: refer this link : https://github.com/sidroopdaska/SelfDrivingCar/blob/master/AdvancedLaneLinesDetection/lane_tracker.ipynb
## TODO: https://github.com/pierluigiferrari/lane_tracker
