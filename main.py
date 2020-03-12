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
		cv2.waitKey(0)#;cv2.destroyAllWindows(i)

                img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		lower_white = np.array([0,0,183])
		upper_white = np.array([255,255,255])
		white_mask = cv2.inRange(img_hsv,lower_white, upper_white)
		cv2.imshow('lane', white_mask);cv2.waitKey(0)
		hist = np.sum(white_mask,axis=0)
		plt.plot(hist)
		plt.show()

		#for i in range(1,len(hist)-1,1):
		#	if hist[i] > hist[i+1] and hist[i]> hist[i-1]:
		#		print(i)

		for i in range(len(hist)):
			if hist[i]:
				white_mask[:,i] = 255
			else :
				white_mask[:,i] = 0
		cv2.imshow('lane_detect', white_mask);cv2.waitKey(0)
		img[:,:,2] = cv2.bitwise_or(img[:,:,2],white_mask)
		img[:,:,0] = cv2.bitwise_and(img[:,:,0],cv2.bitwise_not(white_mask))
		img[:,:,1] = cv2.bitwise_and(img[:,:,1],cv2.bitwise_not(white_mask))
		#cv2.imshow('out',img);cv2.waitKey(0)

		#lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, minLineLength=100, maxLineGap=300)
		## Draw lines on the image
		##imgl = img[int(img.shape[0]):,:,:].copy()
		#for line in lines:
		#	x1, y1, x2, y2 = line[0]
		#	img=cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 3)
		# Show result
		#cv2.imshow("Result Image", img)
		#cv2.waitKey(0);

		org = cv2.warpPerspective(img,invM,(w,h))
		#cv2.imshow('detected Lanes',org);cv2.waitKey(0)
		#cv2.destroyAllWindows()
		org_gray = cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
		ret, org_mask = cv2.threshold(org_gray,10,255,cv2.THRESH_BINARY_INV)
		mask_3d = frame.copy()
		mask_3d[:,:,0] = org_mask
		mask_3d[:,:,1] = org_mask
		mask_3d[:,:,2] = org_mask
		img_masked = cv2.bitwise_and(frame, mask_3d)
		final_image = cv2.add(img_masked, org)
		cv2.imshow('final',final_image);cv2.waitKey(10)

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
