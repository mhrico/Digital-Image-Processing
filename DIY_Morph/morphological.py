import cv2, matplotlib.pyplot as plt, numpy as np

def myOpen(src, kernel):
    temp = myErode(src, kernel)
    temp = myDilate(temp, kernel)
    return temp

def myClose(src, kernel):
    temp = myDilate(src, kernel)
    temp = myErode(temp, kernel)
    return temp

def myDilate(src, kernel):
    output = np.zeros(src.shape, np.uint8)
    k = kernel.shape[0]
    pad = (k-1)//2
    for i in range(pad, src.shape[0] - pad):
        for j in range(pad, src.shape[1] - pad):
            temp = src[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i, j] = np.max(temp * kernel)
    return output

def myErode(src, kernel):
    output = np.zeros(src.shape, np.uint8)
    k = kernel.shape[0]
    pad = (k-1)//2

    for i in range(pad, src.shape[0] - pad):
        for j in range(pad, src.shape[1] - pad):
            temp = src[i-pad:i+pad+1, j-pad:j+pad+1]
            
            output[i,j] = 255 if ((kernel * temp)/255 == kernel).all() else 0
    return output

image = cv2.imread('./image.jpg', 0)
_, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)


kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# open cv er builtin diye test korte ei line lage. Otherwise array er moto korlei hoppe.
dilated = myDilate(binary, kernel)
eroded = myErode(binary, kernel)
opened = myOpen(binary, kernel)
closed = myClose(binary, kernel)

# cv_d = cv2.dilate(binary, kernel)
# cv_e = cv2.erode(binary, kernel)
# cv_op = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# cv_cl = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

output_set = [dilated, eroded, opened, closed] #, cv_d, cv_e, cv_op, cv_cl]

for i in range(len(output_set)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(output_set[i], 'gray')

plt.show()
