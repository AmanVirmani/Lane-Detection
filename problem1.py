import numpy as np
import argparse
import cv2

def main(args):
	cap = cv2.VideoCapture(args['video_path'])
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	if (cap.isOpened() == False):
		print("Error opening video stream or file")
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			img = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
			img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
			enhance_img_hsv = img.copy()
			for i in range(img_hsv.shape[2]):
				enhance_img_hsv[:,:,i] = clahe.apply(img_hsv[:,:,i])
			enhance_img = cv2.cvtColor(enhance_img_hsv,cv2.COLOR_HSV2BGR_FULL)
			enhance_img = cv2.fastNlMeansDenoisingColored(enhance_img,None,10,10,7,21)
			final = np.hstack((img,enhance_img))
			cv2.imshow('out',final);cv2.waitKey(10)
	cv2.destroyAllWindows()

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-cam", "--cam_params", required=False, help="Camera Parameters file for the input data", default='../data/data_1/camera_params.yaml', type=str)
	ap.add_argument("-v", "--video_path", required=False, help="Path for input video", default='../data/NightDrive-2689.mp4', type=str)
	args = vars(ap.parse_args())

	main(args)
