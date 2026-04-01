import cv2
import os
import sys

img_path = sys.argv[1] if len(sys.argv) > 1 else "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/P1/WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000_sharpest_image_1.tif"
if os.path.exists(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        print(f"Image shape: {img.shape}")
        print(f"Data type: {img.dtype}")
        print(f"Size: {img.shape[0]}x{img.shape[1]}")
    else:
        print("Failed to load image")
else:
    print(f"File not found: {img_path}")