# Show effects of some kernels

import cv2 , matplotlib.pyplot as plt, numpy as np

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)

avg = np.ones((3, 3)) / 9

avg_img = cv2.filter2D(image, -1, avg)

plt.subplot(231)
plt.imshow(image, 'gray')
plt.subplot(232)
plt.imshow(avg_img, 'gray')

sobel1 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sob_img1 = cv2.filter2D(image, -1, sobel1)
plt.subplot(233)
plt.imshow(sob_img1, 'gray')

sobel2 = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])
sob_img2 = cv2.filter2D(image, -1, sobel2)
plt.subplot(234)
plt.imshow(sob_img2, 'gray')

laplacian = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

lap_img = cv2.filter2D(image, -1, laplacian)
plt.subplot(235)
plt.imshow(lap_img, 'gray')

gaussian = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

gaus_img = cv2.filter2D(image, -1, gaussian)
plt.subplot(236)
plt.imshow(gaus_img, 'gray')

plt.show()