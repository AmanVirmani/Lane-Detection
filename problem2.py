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

def readCamParams(fname):
	with open(fname) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	K = np.array([ float(num) for num in params['K'].split()])
	D = np.array([ float(num) for num in params['D'].split()])
	K = K.reshape(3,3)
	return K,D

def processImage(img,K,D) :
	img = removeDistortion(img, K, D)
	dst = img;#cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
	return dst

def warpImage(img,src_pts,dst_pts):
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	invM = cv2.getPerspectiveTransform(dst_pts, src_pts)
	img = cv2.warpPerspective(img, M, (300, 300))
	return img, M, invM

def getLaneMask(img):
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([0,52,127])
	upper_yellow = np.array([26,156,255])
	yellow_mask = cv2.inRange(img_hsv,lower_yellow, upper_yellow)

	lower_white = np.array([0,0,183])
	upper_white = np.array([255,255,255])
	white_mask = cv2.inRange(img_hsv,lower_white, upper_white)

	mask = cv2.bitwise_or(white_mask,yellow_mask)
	return mask

def detectLanes(img, hist,mask):
	midpoint = np.int(hist.shape[0] / 2)
	leftx_base = np.argmax(hist[:midpoint])
	rightx_base = np.argmax(hist[midpoint:]) + midpoint

	out = np.dstack((mask, mask, mask)) #* 255
	#cv2.imshow('out',out);cv2.waitKey(0)

	nb_windows = 12 # number of sliding windows
	margin = 10 # width of the windows +/- margin
	minpix = 5 # min number of pixels needed to recenter the window
	window_height = int(mask.shape[0] / nb_windows)
	min_lane_pts = 10  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a

	nonzero = mask.nonzero()
	nonzerox = np.array(nonzero[1])
	nonzeroy = np.array(nonzero[0])

	leftx_current = leftx_base
	rightx_current = rightx_base

	left_lane_inds = []
	right_lane_inds = []

	for window in range(nb_windows):
		# Identify window boundaries in x and y (and left and right)
		win_y_low = mask.shape[0] - (1 + window) * window_height
		win_y_high = mask.shape[0] - window * window_height

		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin

		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw windows for visualisation
		cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),\
					  (0, 255, 0), 2)
		cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),\
					  (0, 255, 0), 2)

		#cv2.imshow('out',out);cv2.waitKey(0)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
						 & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
						 & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) >  minpix:
			leftx_current = int(np.mean(nonzerox[good_left_inds]))

		if len(good_right_inds) > minpix:
			rightx_current = int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	## Extract pixel positions for the left and right lane lines
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_fit, right_fit = None, None

	## Sanity check; Fit a 2nd order polynomial for each lane line pixels
	if len(leftx) >= min_lane_pts and len(rightx) >= min_lane_pts:
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

	if left_fit is None or right_fit is None:
		print('no fit')
		return
	plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(img, left_fit, right_fit)

	## Color the detected pixels for each lane line
	final_mask = np.zeros(mask.shape)
	final_mask[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255]
	final_mask[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255]
	#final_mask[plot_yleft, left_lane_inds] = [255]
	#final_mask[plot_yright, right_lane_inds] = [255]
	out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 10, 255]

	left_poly_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
	right_poly_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])

	## Plot the fitted polynomial
	cv2.polylines(out, np.int32([left_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
	cv2.polylines(out, np.int32([right_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
	cv2.imshow('out',out);cv2.waitKey(0)

	if (left_fit[1] + right_fit[1]) / 2 > 0.05:
		text = 'Left turn'
	elif (left_fit[1] + right_fit[1]) / 2 < -0.05:
		text = 'Right turn'
	else:
		text = 'Straight'
	print(text)
	return final_mask.astype(np.uint8)


def plotLanes(src, img, mask,invM):
	hist = np.sum(mask,axis=0)
	mask = detectLanes(img, hist,mask)
	if mask is None:
		return None
	cv2.imshow('final mask',mask);cv2.waitKey(0)
	hist = np.sum(mask,axis=0)
	for i in range(len(hist)):
		if hist[i]:
			mask[:,i] = 255
		else :
			mask[:,i] = 0

	#ret,out,poly = polyfit_sliding_window(mask,visualise=True)
	img[:,:,2] = cv2.bitwise_or(img[:,:,2],mask)
	img[:,:,0] = cv2.bitwise_and(img[:,:,0],cv2.bitwise_not(mask))
	img[:,:,1] = cv2.bitwise_and(img[:,:,1],cv2.bitwise_not(mask))
	#cv2.imshow('out',img);cv2.waitKey(0)

	org = cv2.warpPerspective(img,invM,(src.shape[1],src.shape[0]))
	org_gray = cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
	ret, org_mask = cv2.threshold(org_gray,10,255,cv2.THRESH_BINARY_INV)
	mask_3d = src.copy()
	mask_3d[:,:,0] = org_mask
	mask_3d[:,:,1] = org_mask
	mask_3d[:,:,2] = org_mask
	img_masked = cv2.bitwise_and(src, mask_3d)
	final_image = cv2.add(img_masked, org)
	return final_image


def main(args):
	src_pts = np.float32([
		(200,511),
		(600,277),
		(723,277),
		(900,511)
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
		frame =img.copy()
		h,w = img.shape[:2]
		#cv2.imshow('before',img); cv2.waitKey(0);
		dst = processImage(img,K,D)
		#cv2.imshow('after denoising',dst); cv2.waitKey(0);

		img, M, invM = warpImage(img,src_pts,dst_pts)
		cv2.imshow("bird's eye", img); cv2.waitKey(0)

		mask = getLaneMask(img)
		cv2.imshow('lane', mask);cv2.waitKey(0)
		final_image = plotLanes(frame, img, mask, invM)
		if final_image is None:
			continue
		cv2.imshow('final', final_image); cv2.waitKey(10)

		#break

def get_poly_points(img, left_fit, right_fit):
	'''
	Get the points for the left lane/ right lane defined by the polynomial coeff's 'left_fit'
	and 'right_fit'
	:param left_fit (ndarray): Coefficients for the polynomial that defines the left lane line
	:param right_fit (ndarray): Coefficients for the polynomial that defines the right lane line
	: return (Tuple(ndarray, ndarray, ndarray, ndarray)): x-y coordinates for the left and right lane lines
	'''
	ysize, xsize = img.shape[:2]

	# Get the points for the entire height of the image
	plot_y = np.linspace(0, ysize-1, ysize)
	plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
	plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

	# But keep only those points that lie within the image
	plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
	plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
	plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
	plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))

	return plot_xleft.astype(np.int), plot_yleft.astype(np.int), plot_xright.astype(np.int), plot_yright.astype(np.int)

def main2(args):
	src_pts = np.float32([
		(0,719),
		(537,489),
		(759,489),
		(1204,719)
	])

	dst_pts = np.float32([
		(0,300),
		(0,0),
		(300,0),
		(300,300)
	])
	cap = cv2.VideoCapture(args['video_path'])
	K,D = readCamParams(args['cam_params'])
	if (cap.isOpened() == False):
		print("Error opening video stream or file")
	while (cap.isOpened()):
		ret,frame = cap.read()
		if ret:
			dst = processImage(frame.copy(),K,D)

			img, M, invM = warpImage(dst, src_pts, dst_pts)
			cv2.imshow("bird's eye",img); cv2.waitKey(0)

			mask = getLaneMask(img)
			cv2.imshow('lane_detect', mask);cv2.waitKey(0)

			final_image = plotLanes(frame,img, mask, invM)
			if final_image is None:
				continue
			cv2.imshow('final',final_image);cv2.waitKey(0)
		else:
			break

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-cam", "--cam_params", required=False, help="Camera Parameters file for the input data", default='../data/data_1/camera_params.yaml', type=str)
	ap.add_argument("-i", "--image_path", required=False, help="Path for input images", default='../data/data_1/data/', type=str)
	ap.add_argument("-v", "--video_path", required=False, help="Path for input video", default='../data/data_2/challenge_video.mp4', type=str)
	args = vars(ap.parse_args())

	main(args)
	#main2(args)

