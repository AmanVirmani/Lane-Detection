import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Night Drive - 2689.mp4')
count = 0
while(cap.isOpened()):
    count +=  1
    if count < 100:
        continue
    ret, frame = cap.read()
    
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.show()
    H, S, V = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    
    eq_V = cv2.equalizeHist(V)
    eq_hsv = cv2.merge([H, S, eq_V])
    eq_image = cv2.cvtColor(eq_hsv,cv2.COLOR_HSV2RGB)
    plt.imshow(eq_image)
    plt.show()
    #cv2.waitKey(0);
    #cv2.destroyAllWindows()
    break