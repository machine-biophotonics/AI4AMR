#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os

os.makedirs('crop_visualizations', exist_ok=True)

img = np.zeros((2720, 2720, 3), dtype=np.uint8)

for i in range(2720):
    for j in range(2720):
        img[i, j] = [
            int(128 + 50 * np.sin(i/100)),
            int(128 + 50 * np.cos(j/100)),
            int(128 + 50 * np.sin((i+j)/150))
        ]

for cx in range(400, 2400, 400):
    for cy in range(400, 2400, 400):
        for i in range(-100, 100):
            for j in range(-100, 100):
                if i*i + j*j < 80*80:
                    if 0 <= cx+i < 2720 and 0 <= cy+j < 2720:
                        img[cy+j, cx+i] = [200, 180, 150]

Image.fromarray(img).save('crop_visualizations/sample_image.tif')
print('Created sample_image.tif')