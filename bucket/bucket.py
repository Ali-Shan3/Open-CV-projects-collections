import cv2
import numpy as np

# --- Load images (grayscale already) ---
empty_img = cv2.imread("E.jpg", cv2.IMREAD_GRAYSCALE)
filled_img = cv2.imread("PF1.jpg", cv2.IMREAD_GRAYSCALE)

# --- Step 1: Find bucket contour from empty image ---
blur = cv2.GaussianBlur(empty_img, (5, 5), 0)
_, th_empty = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean up noise with morphology
kernel = np.ones((5, 5), np.uint8)
th_empty = cv2.morphologyEx(th_empty, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(th_empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bucket_contour = max(contours, key=cv2.contourArea)

# Smooth the contour
epsilon = 0.01 * cv2.arcLength(bucket_contour, True)
bucket_contour = cv2.approxPolyDP(bucket_contour, epsilon, True)

# Create bucket mask
mask = np.zeros_like(empty_img)
cv2.drawContours(mask, [bucket_contour], -1, 255, -1)

# --- Step 2: Difference between empty and filled ---
diff = cv2.absdiff(filled_img, empty_img)
diff_masked = cv2.bitwise_and(diff, diff, mask=mask)

# --- Step 3: Threshold to detect objects inside bucket ---
_, th = cv2.threshold(diff_masked, 25, 255, cv2.THRESH_BINARY)

# --- Step 4: Calculate fullness percentage ---
filled_pixels = cv2.countNonZero(th)
total_pixels = cv2.countNonZero(mask)
fullness = (filled_pixels / total_pixels) * 100
print(f"Bucket fullness: {fullness:.2f}%")

# --- Step 5: Visualize results ---
result_vis = cv2.cvtColor(filled_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_vis, [bucket_contour], -1, (0, 255, 0), 2)
cv2.putText(result_vis, f"Fullness: {fullness:.1f}%", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Convert threshold (1 channel) to BGR so we can stack with result
th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

# Resize both to the same height if needed
h = min(result_vis.shape[0], th_bgr.shape[0])
result_vis = cv2.resize(result_vis, (int(result_vis.shape[1] * h / result_vis.shape[0]), h))
th_bgr = cv2.resize(th_bgr, (result_vis.shape[1], h))

# Stack side by side
combined = np.hstack((result_vis, th_bgr))

cv2.imshow("Result + Threshold", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

