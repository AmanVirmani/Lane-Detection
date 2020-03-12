# Author Moumita Paul

import numpy as np
import cv2

# Reading Video
cap = cv2.VideoCapture('Night Drive - 2689.mp4')

                # ---------------#
                #   writing video
                #----------------#
# Default resolution 
# frame_width= int(cap.get(3))
# frame_height=int(cap.get(4))

# # video write object Creation
# output=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()

    # Convert RGB image to YUV image
    img_yuv= cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    if ret == True:
        # cv2.imshow('Color Input Image',frame )

        cv2.waitKey(0); # & 0xFF == ord('q'):

        # Equalize the Histogram of Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # Convert the YUV image back to RGB format
        img_output= cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
        
        cv2.imshow('Histogram Equalized',img_output)

        # output.write(img_output)

        cv2.waitKey(0);

        # Image Save
        cv2.imwrite('1st_frame_output.png',img_output)
        cv2.destroyAllWindows()
        
        break