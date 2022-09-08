# slice planes of image

import cv2, matplotlib.pyplot as plt

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)

planes = []

for i in range(8):
    planes.append(image & 2 ** i)
    plt.subplot(2, 4, i+1)
    plt.imshow(planes[i], 'gray')

rec = planes[6] + planes[7] + planes[5] + planes[4]
plt.figure()
plt.imshow(rec, 'gray')

plt.show()