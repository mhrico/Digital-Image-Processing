import cv2, matplotlib.pyplot as plt, numpy as np


image = cv2.imread('./image.jpg', 0)
hist = cv2.calcHist([image], [0], None, [256], [0,256])

plt.subplot(221)
plt.imshow(image, 'gray')
plt.subplot(222)
plt.plot(hist)

cdf = hist.cumsum()
cdfMin = cdf.min()
sz = image.size
equalized = np.zeros(image.shape, np.uint8)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        equalized[i,j] = ((cdf[image[i,j]] - cdfMin) / (sz - cdfMin)) * (256-1)


equalizedHist = cv2.calcHist([equalized], [0], None, [256], [0,256])
plt.subplot(223)
plt.imshow(equalized, 'gray')
plt.subplot(224)
plt.plot(equalizedHist)

plt.show()