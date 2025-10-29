from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (244, 244)})
picam2.configure(config)
picam2.start()
time.sleep(2)  # Allow the camera to warm up

def capture_frame():
    frame = picam2.capture_array()
    frame = np.rot90(frame, k=2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def save_image(frame, path):
    import cv2
    cv2.imwrite(path, frame)
