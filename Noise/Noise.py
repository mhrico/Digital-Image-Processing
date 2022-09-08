# Show effects of salt and pepper noise and denoise using avg, gaussian and median

from audioop import avg
import cv2, matplotlib.pyplot as plt, numpy as np

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
noised_img = image.copy()

for i in range(3000):
    y = np.random.randint(0, noised_img.shape[0])
    x = np.random.randint(0, noised_img.shape[1])
    noised_img[x, y] = np.random.randint(0, 2) * 255

gaus_img = cv2.GaussianBlur(noised_img, (3,3), 1)
avg_img = cv2.blur(noised_img, (3,3))
med_img = cv2.medianBlur(noised_img, 3)

img_set = [image, noised_img, gaus_img, avg_img, med_img]
for i in range(len(img_set)):
    plt.subplot(1, 5, i+1)
    plt.imshow(img_set[i], 'gray')

plt.show()