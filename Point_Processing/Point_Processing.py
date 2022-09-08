'''
Apply point processing
s = 100, if r >= T1 and r <= T2; otherwise
           s = 10.

s = 100, if r >= T1 and r <= T2; otherwise
           s = r.

s = c log(1 + r) .

s = c ( r + epsilon ) ^ p
'''

import math
import cv2
import matplotlib.pyplot as plt

def show(image, newImg):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image, 'gray')
    plt.subplot(122)
    plt.imshow(newImg, 'gray')

def main():

    image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
    newImg = image.copy()

    t1 = 50
    t2 = 100

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newImg[i, j] = 100 if newImg[i, j] >= t1 and newImg[i, j] <= t2 else 10

    show(image, newImg)

    newImg = image.copy()
    c = 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newImg[i, j] = c * math.log((1 + newImg[i, j]))

    show(image, newImg)
    plt.show()

if __name__ == '__main__':
    main()