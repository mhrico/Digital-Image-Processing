from unicodedata import name
import matplotlib.pyplot as plt
import numpy as np
import cv2
def main():
    image = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
    ft_img = np.fft.fft2(image)

    # for showing
    ft_abs = np.log(np.abs(ft_img))
    ft_abs_shifted = np.log(np.abs(np.fft.fftshift(ft_img)))
    # for showing

    k = cv2.getGaussianKernel(9, 0)
    kernel = k @ k.T
    ft_kernel = np.fft.fft2(kernel, image.shape)

    # for showing
    ft_kernel_abs = np.abs(ft_kernel)
    ft_kernel_abs_shifted = np.abs(np.fft.fftshift(ft_kernel))
    # for showing

    freq_filtered = ft_img * ft_kernel

    # for showing
    filtered_abs_shifted = np.abs(ft_kernel_abs_shifted * ft_abs_shifted)
    # for showing

    sp_dom_filtered = np.real(np.fft.ifft2(freq_filtered))

    output_set = [image, ft_abs,ft_abs_shifted, kernel, ft_kernel_abs, ft_kernel_abs_shifted, filtered_abs_shifted, sp_dom_filtered]
    title_set = ['Image', 'FT Magnitude', 'FT Mag. Shifted', 'Gaussian Kernel', 'Kernel FT', 'Kernel FT Shifted', 'Filtered FT Shifted', 'Spatial Domain Result']

    for i in range(len(output_set)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(output_set[i], 'gray')
        plt.title(title_set[i])
        plt.imsave('./output/{}.jpg'.format(title_set[i]), output_set[i], cmap = 'gray')

    plt.show()

if __name__ == '__main__':
    main()