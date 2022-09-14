import cv2
import matplotlib.pyplot as plt
import numpy as np

def dilation(source, kernel):
    k = kernel.shape[0]
    output = np.zeros(source.shape, source.dtype)
    pad = (k-1)//2

    for i in range(pad, source.shape[0] - pad):
        for j in range(pad, source.shape[1] - pad):
            temp = source[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i, j] = (temp * kernel).max()
    return output

def erosion(source, kernel):
    k = kernel.shape[0]
    output = np.zeros(source.shape, source.dtype)
    pad = (k-1)//2

    for i in range(pad, source.shape[0] - pad):
        for j in range(pad, source.shape[1] - pad):
            temp = source[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i, j] = 255 if ((temp * kernel)/255 == kernel).all() else 0
    return output

def open(source, kernel):
    temp = erosion(source, kernel)
    temp = dilation(temp, kernel)
    return temp

def close(source, kernel):
    temp = dilation(source, kernel)
    temp = erosion(temp, kernel)
    return temp

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

md = dilation(binary, kernel)
cd = cv2.dilate(binary, kernel)
me = erosion(binary, kernel)
ce = cv2.erode(binary, kernel)
mo = open(binary, kernel)
mc = close(binary, kernel)
co = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cc = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

outset = [binary, md, cd, me, ce, mo, co, mc, cc]

for i in range(len(outset)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(outset[i], 'gray')

plt.show()
