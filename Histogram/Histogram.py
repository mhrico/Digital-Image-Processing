from msilib.schema import Binary
import cv2, matplotlib.pyplot as plt

image = plt.imread('./image.jpg')

r, g, b = cv2.split(image)

grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

img_set = [image, r, g, b, grayscale, binary]

plt.figure()
for i in range(len(img_set)):
    plt.subplot(2, 3, i+1)
    if len(img_set[i].shape) == 3:
        plt.imshow(img_set[i])
    else:
        plt.imshow(img_set[i], 'gray')

# histogram using matplotlib
plt.figure()
for i in range(1, len(img_set)):
    plt.subplot(2, 3, i)
    plt.hist(img_set[i].ravel(), 256, (0, 255))

# histogram using open-cv

plt.figure()
for i in range(1, len(img_set)-2):
    h = cv2.calcHist([img_set[i]], [0], None, [256], [0, 256])
    plt.plot(h)

plt.show()
