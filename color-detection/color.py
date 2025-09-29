import cv2
import numpy as np

from util import get_limits  # keep your existing function

# Pick target color (yellow in BGR)
yellow = [0, 255, 255]

cap = cv2.VideoCapture(0)  # use 0 unless you know your cam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV range for selected color
    lowerLimit, upperLimit = get_limits(color=yellow)

    # Create mask for chosen color
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Extract the colored region
    color_part = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert whole frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Combine color + gray
    highlighted = cv2.addWeighted(color_part, 1, gray_bgr, 1, 0)

    cv2.imshow("Smart Highlighter", highlighted)
    cv2.imshow("Mask", mask)  # optional debug window

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
