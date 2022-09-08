# shift the histogram and see what happens

import cv2, matplotlib.pyplot as plt, numpy as np

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
rs_img = image.copy()
ls_img = image.copy()
rng_img = np.clip(image, 50, 100)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        rs_img[i, j] = np.clip((rs_img[i, j] + 100), 0, 255)
        ls_img[i, j] = np.clip((ls_img[i, j] - 100), 0, 255)

img_set = [image, rs_img, ls_img, rng_img]

for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(img_set[i], 'gray')
    plt.subplot(2, 4, i + 1 + 4)
    plt.hist(img_set[i].ravel(), 256, (0, 256))

plt.show()