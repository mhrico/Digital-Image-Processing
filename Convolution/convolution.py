import cv2, matplotlib.pyplot as plt, numpy as np

def convolve(src, kernel):
    k = kernel.shape[0]
    padding = (k-1)//2
    output = np.zeros(src.shape, np.float64)
    src = cv2.copyMakeBorder(src, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, src.shape[0] - padding):
        for j in range(padding, src.shape[1] - padding):
            temp = src[i-padding:i+padding+1, j-padding:j+padding+1]
            output[i-padding, j-padding] = np.sum(np.multiply(temp, kernel))
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)
    return output

image = cv2.imread('./image.jpg', 0)

horizonal = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
box = np.ones((7,7)) / 49
h_edge = convolve(image, horizonal)
v_edge = convolve(image, vertical)
blurry = convolve(image, box)
cv_h = cv2.filter2D(image, -1, horizonal)
cv_v = cv2.filter2D(image, -1, vertical)
cv_blur = cv2.filter2D(image, -1, box)

output_set = [h_edge, v_edge, blurry, cv_h, cv_v, cv_blur]

for i in range(len(output_set)):
    plt.subplot(2, 3, i +1)
    plt.imshow(output_set[i], 'gray')

plt.show()
