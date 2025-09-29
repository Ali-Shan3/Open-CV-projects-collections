import cv2

# ---------------- Tracker Factory (compatible with all versions) ----------------
def create_tracker(tracker_type="MOSSE"):
    # Get OpenCV version
    (major, minor, subminor) = cv2.__version__.split(".")
    major, minor = int(major), int(minor)

    if tracker_type.upper() == "MOSSE":
        if major == 3 and hasattr(cv2, "TrackerMOSSE_create"):  # OpenCV 3.x
            return cv2.TrackerMOSSE_create()
        elif major >= 4:
            if hasattr(cv2, "legacy"):  # OpenCV 4.5.1+
                return cv2.legacy.TrackerMOSSE_create()
            elif hasattr(cv2, "TrackerMOSSE_create"):  # Some 4.x builds
                return cv2.TrackerMOSSE_create()

    elif tracker_type.upper() == "CSRT":
        if major == 3 and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif major >= 4:
            if hasattr(cv2, "legacy"):
                return cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()

    elif tracker_type.upper() == "KCF":
        if major == 3 and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        elif major >= 4:
            if hasattr(cv2, "legacy"):
                return cv2.legacy.TrackerKCF_create()
            elif hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()

    raise ValueError(f"Tracker type {tracker_type} not supported on OpenCV {cv2.__version__}")

# ---------------- Initialize Tracker ----------------
tracker = create_tracker("MOSSE")  # You can switch to "CSRT" or "KCF"

cap = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

# Read first frame
success, frame = cap.read()
if not success:
    print("Failed to read from camera")
    exit()

# Select ROI for tracking
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

# ---------------- Helper Function ----------------
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3, 3)
    cv2.putText(img, "Tracking", (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ---------------- Main Loop ----------------
while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    if not success:
        break

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (100, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fps > 60:
        myColor = (20, 230, 20)
    elif fps > 20:
        myColor = (230, 20, 20)
    else:
        myColor = (20, 20, 230)

    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(img, "Fps:", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, str(int(fps)), (75, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)
    cv2.putText(img, "Status:", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Tracking", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
