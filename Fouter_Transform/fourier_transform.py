import matplotlib.pyplot as plt
import numpy as np
import cv2

def kernel_transformer(kernel):
    size = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])
    kernel = np.pad(kernel, (((size[0]+1)//2, size[0]//2), ((size[1]+1)//2, size[1]//2)), 'constant')
    kernel = np.fft.ifftshift(kernel)
    return kernel


img = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
transformed_img = np.fft.fft2(img)
transformed_img_magnitude = 100 * np.log(np.abs(transformed_img))
centered_transformed_img_magnitude = 100 * np.log(np.abs(np.fft.fftshift(transformed_img)))

# average filter
avg_kernel = np.ones((3,3)) / 9
transformed_avg = kernel_transformer(avg_kernel)
avg_filtered_img = np.real(np.fft.ifft2(transformed_img * np.fft.fft2(transformed_avg)))

# sobel filter

sob_kernelX = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

sob_kernelY = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

transformed_sobelX = kernel_transformer(sob_kernelX)
transformed_sobelY = kernel_transformer(sob_kernelY)

sobelX_filtered_img = np.real(np.fft.ifft2(transformed_img * np.fft.fft2(transformed_sobelX)))
sobelY_filtered_img = np.real(np.fft.ifft2(transformed_img * np.fft.fft2(transformed_sobelY)))

# laplacian filter
lap_kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

transformed_lap = kernel_transformer(lap_kernel)
lap_filtered_image = np.real(np.fft.ifft2(transformed_img * np.fft.fft2(transformed_lap)))


image_set = [img, transformed_img_magnitude, centered_transformed_img_magnitude, avg_filtered_img, sobelX_filtered_img, sobelY_filtered_img, lap_filtered_image]
title_set = ['Grayscale', 'F.T. Magnitude', 'Centered F.T. Mag.', 'Average Filter', 'Sobel X-Axis Filter', 'Sobel Y-Axis Filter', 'Laplacian Filter']

j = len(title_set)
plt.figure('Output', (10,6))
for i in range(j):
    plt.subplot(2, 4, i + 1)
    plt.imshow(image_set[i], cmap='gray')
    plt.title(title_set[i])
    plt.imsave('./' + title_set[i] + '.png', image_set[i], cmap='gray')
plt.tight_layout()
plt.savefig('./Output.png')
plt.show()