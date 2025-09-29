import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (replace 'lena.png' with your photo)
img = cv2.imread("3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) → RGB (Matplotlib)

# --- Define Filters ---
# 1. Blur
blur = cv2.blur(img, (5, 5))

# 2. Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Median Blur
median = cv2.medianBlur(img, 5)

# 4. Bilateral Filter (edge-preserving)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# 5. Sharpen (using kernel)
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpen = cv2.filter2D(img, -1, kernel_sharpen)

# --- Display Original vs Filtered ---
filters = {
    "Original": img,
    "Blur": blur,
    "Gaussian Blur": gaussian,
    "Median Blur": median,
    "Bilateral Filter": bilateral,
    "Sharpen": sharpen
}

plt.figure(figsize=(12, 8))

for i, (name, filtered_img) in enumerate(filters.items()):
    plt.subplot(2, 3, i + 1)
    plt.imshow(filtered_img)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
