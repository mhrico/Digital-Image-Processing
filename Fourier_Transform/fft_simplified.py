import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
fft_img = np.fft.fft2(image)
im_abs = np.log(np.abs(fft_img))
im_shifted = np.log(np.abs(np.fft.fftshift(fft_img)))

# avg_kernel = np.ones((5,5)) / 25
avg_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
fft_avg_k = np.fft.fft2(avg_kernel, image.shape)
avg_k_abs = np.log(np.abs(fft_avg_k))
ak_shifted = np.abs(np.fft.fftshift(fft_avg_k))

freq_filtered = fft_avg_k * fft_img

sp_dom_filtered = np.real(np.fft.ifft2(freq_filtered))


out_set = [image, im_abs,im_shifted, avg_kernel, avg_k_abs, ak_shifted, sp_dom_filtered]

for i in range(len(out_set)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(out_set[i], 'gray')

plt.show()