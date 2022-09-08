# equalize histogram of an image

import cv2, matplotlib.pyplot as plt

image = cv2.imread('./image.jpg', 0)

plt.subplot(221)
plt.imshow(image, 'gray')
plt.subplot(222)
plt.hist(image.ravel(), 256, (0, 256))

equalized = cv2.equalizeHist(image)
plt.subplot(223)
plt.imshow(equalized, 'gray')
plt.subplot(224)
plt.hist(equalized.ravel(), 256, (0, 256))

plt.show()