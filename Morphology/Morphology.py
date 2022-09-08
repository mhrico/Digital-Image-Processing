# apply erosion, dilation, opening and closing

import cv2, matplotlib.pyplot as plt, numpy as np

image = cv2.imread('./image.jpg', 0)

_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3))

eroded = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

img_set = [image, binary, eroded, dilated, opened, close]

for i in range(len(img_set)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_set[i], 'gray')

plt.show()