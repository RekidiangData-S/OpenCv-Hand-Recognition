from detect_hand import *
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
# use this to capture a live histogram
hist = capture_histogram(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hand = detect_hand(frame, hist)

    
    cv2.imshow("Raw", hand["raw"])
    cv2.imshow("Enhanced Binary", hand["binary"])
    cv2.imshow("Masked", hand["masked"])



    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
