import cv2
import numpy as np

# --- Load images (grayscale already) ---
empty_img = cv2.imread("E.jpg", cv2.IMREAD_GRAYSCALE)
filled_img = cv2.imread("E.jpg", cv2.IMREAD_GRAYSCALE)

# --- Step 1: Find bucket contour from empty image ---
blur = cv2.GaussianBlur(empty_img, (5, 5), 0)
_, th_empty = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(th_empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bucket_contour = max(contours, key=cv2.contourArea)   # largest contour = bucket

# Create bucket mask
mask = np.zeros_like(empty_img)
cv2.drawContours(mask, [bucket_contour], -1, 255, -1)

# --- Step 2: Difference between empty and filled ---
diff = cv2.absdiff(filled_img, empty_img)
diff_masked = cv2.bitwise_and(diff, diff, mask=mask)

# --- Step 3: Threshold to highlight cookies ---
_, th = cv2.threshold(diff_masked, 25, 255, cv2.THRESH_BINARY)

# --- Step 4: Calculate fullness percentage ---
filled_pixels = cv2.countNonZero(th)
total_pixels = cv2.countNonZero(mask)
fullness = (filled_pixels / total_pixels) * 100

print(f"Bucket fullness: {fullness:.2f}%")

# --- Step 5: Visualize results ---
result_vis = cv2.cvtColor(filled_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_vis, [bucket_contour], -1, (0, 255, 0), 2)  # bucket outline
cv2.putText(result_vis, f"Fullness: {fullness:.1f}%", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Detected Bucket (Contour)", result_vis)
cv2.imshow("Thresholded Difference", th)
cv2.waitKey(0)
cv2.destroyAllWindows()
