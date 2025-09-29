import cv2

# --- Paths of input images ---
image_paths = [
    '1.png',
    '2.png',
    '3.png'
]

# --- Initialize a list of images ---
imgs = []

# --- Load and resize images ---
for i in range(len(image_paths)):
    img = cv2.imread(image_paths[i])
    if img is None:
        print(f"Error: Could not load {image_paths[i]}")
        continue
    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
    imgs.append(img)

# --- Show input images ---
for idx, im in enumerate(imgs):
    cv2.imshow(f'Image {idx+1}', im)

# --- Stitcher object ---
stitchy = cv2.Stitcher.create()

# Perform stitching
(status, output) = stitchy.stitch(imgs)

# --- Check result ---
if status != cv2.STITCHER_OK:
    print("Stitching ain't successful, error code =", status)
else:
    print("Your Panorama is ready!!!")
    cv2.imshow('Final Result', output)
    cv2.imwrite('Result-Image', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
