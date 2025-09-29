# ar_billboard.py
import cv2
import numpy as np
import time

##############################
# Config / file paths
##############################
REF_IMG_PATH = "bb.jpeg"   # A clean reference photo of the billboard area
OVERLAY_PATH = "1.jpg"          # The image you want to paste (supports PNG with alpha)
MIN_MATCHES = 10                     # Minimum good matches to accept homography
SMOOTH_ALPHA = 0.65                  # Smoothing factor for polygon (0..1) higher = smoother/laggier

##############################
# Helper functions
##############################
def load_overlay(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Overlay not found: {path}")
    if img.shape[2] == 4:
        overlay_rgb = img[:, :, :3]
        overlay_alpha = img[:, :, 3]
    else:
        overlay_rgb = img
        overlay_alpha = None
    return overlay_rgb, overlay_alpha

def order_points_clockwise(pts):
    # pts: Nx2 array
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # tr
    rect[3] = pts[np.argmax(diff)] # bl
    return rect

def warp_overlay_to_frame(overlay_rgb, overlay_alpha, H, frame_shape):
    h_f, w_f = frame_shape[:2]
    # warp rgb
    warped_rgb = cv2.warpPerspective(overlay_rgb, H, (w_f, h_f))
    if overlay_alpha is not None:
        warped_alpha = cv2.warpPerspective(overlay_alpha, H, (w_f, h_f))
        mask = (warped_alpha > 10).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2GRAY)
        mask = (gray > 5).astype(np.uint8) * 255
    return warped_rgb, mask

def blend_images(frame, warped_rgb, mask):
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    warped_f = warped_rgb.astype(np.float32)
    frame_f = frame.astype(np.float32)
    blended = (frame_f * (1.0 - mask_f) + warped_f * mask_f).astype(np.uint8)
    return blended

def find_quad_by_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    area_thresh = (w*h) * 0.02
    best_quad = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx.reshape(4,2).astype(np.float32)
    if best_quad is not None:
        return order_points_clockwise(best_quad)
    return None

##############################
# Initialization
##############################
# Load reference image and compute keypoints/descriptors
ref_img = cv2.imread(REF_IMG_PATH)
if ref_img is None:
    raise FileNotFoundError(f"Reference image not found: {REF_IMG_PATH}")
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
ref_h, ref_w = ref_img.shape[:2]

# Load overlay
overlay_rgb, overlay_alpha = load_overlay(OVERLAY_PATH)

# Ensure overlay is same aspect as ref (optional): resize overlay to ref dimensions
overlay_rgb = cv2.resize(overlay_rgb, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
if overlay_alpha is not None:
    overlay_alpha = cv2.resize(overlay_alpha, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)

# ORB detector
orb = cv2.ORB_create(nfeatures=2000)

kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
if des_ref is None or len(kp_ref) < 8:
    raise RuntimeError("Not enough features in reference image. Use a richer reference photo.")

# BFMatcher with Hamming (for ORB)
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

# Corners of reference image
ref_corners = np.float32([[0,0], [ref_w,0], [ref_w,ref_h], [0,ref_h]]).reshape(-1,1,2)

# For smoothing / fallback
prev_polygon = None
prev_H = None
prev_time = time.time()
fallback_frames = 0
MAX_FALLBACK = 10  # keep using last good H for up to 10 frames if matching is lost

# Video capture (camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("Press ESC or 'q' to quit")

##############################
# Main loop
##############################
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

        H = None
        polygon_pts = None

        if des_frame is not None and len(kp_frame) > 8:
            # Match descriptors
            matches = bf.knnMatch(des_ref, des_frame, k=2)
            # Lowe ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) >= MIN_MATCHES:
                src_pts = np.float32([ kp_ref[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    # Map ref corners to frame to get polygon
                    dst_corners = cv2.perspectiveTransform(ref_corners, H).reshape(4,2)
                    polygon_pts = order_points_clockwise(dst_corners)
                    prev_H = H
                    fallback_frames = 0
            else:
                # not enough matches
                H = None

        # Fallback 1: use previous homography if available (briefly)
        if polygon_pts is None and prev_H is not None and fallback_frames < MAX_FALLBACK:
            dst_corners = cv2.perspectiveTransform(ref_corners, prev_H).reshape(4,2)
            polygon_pts = order_points_clockwise(dst_corners)
            H = prev_H
            fallback_frames += 1

        # Fallback 2: contour / quad detection if feature matching failed entirely
        if polygon_pts is None:
            quad = find_quad_by_contours(frame)
            if quad is not None:
                polygon_pts = quad
                # Build homography from ref image to detected quad
                H, _ = cv2.findHomography(ref_corners.reshape(-1,2), polygon_pts.reshape(-1,2), cv2.RANSAC, 5.0)
                prev_H = H
                fallback_frames = 0

        output = frame.copy()

        if H is not None and polygon_pts is not None:
            # Smooth polygon (exponential moving average)
            if prev_polygon is None:
                smooth_poly = polygon_pts
            else:
                smooth_poly = SMOOTH_ALPHA * prev_polygon + (1.0 - SMOOTH_ALPHA) * polygon_pts
            prev_polygon = smooth_poly

            # Warp overlay to frame
            warped_rgb, mask = warp_overlay_to_frame(overlay_rgb, overlay_alpha, H, frame.shape)
            # Blend
            blended = blend_images(output, warped_rgb, mask)
            output = blended

            # Draw polygon contour
            pts_draw = np.round(smooth_poly).astype(int)
            cv2.polylines(output, [pts_draw], True, (0,255,0), 3, cv2.LINE_AA)
            # optionally fill small circle on corners
            for p in pts_draw:
                cv2.circle(output, tuple(p), 5, (0,0,255), -1)
            cv2.putText(output, f"Detected (matches used)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            prev_polygon = None
            cv2.putText(output, "Billboard not detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # show
        cv2.imshow("AR Billboard - press q or ESC to exit", output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
