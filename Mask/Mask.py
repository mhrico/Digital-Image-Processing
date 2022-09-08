# Apply a binary mask to an image

import cv2, matplotlib.pyplot as plt, numpy as np

image = cv2.imread('./image.jpg', 0)

mask = np.zeros(image.shape, 'uint8')
cv2.circle(mask, (127, 127), 50, 255, cv2.FILLED)

masked_img = image & mask

img_set = [image, mask, masked_img]
for i in range(len(img_set)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img_set[i], 'gray')
    
plt.show()