from extract_fingertrip import *
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
    if hand_detected:
        hand_image = hand["boundaries"]
        fingertrips = extract_fingertrips(hand)
        plot(hand_image, fingertrips)
        cv2.imshow("Hand Detector", hand_image)

    else:
        cv2.imshow("Hand Detector", frame)

    
    
    k = cv2.waitKey(30)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
