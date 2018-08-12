
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift
from skimage import data
from skimage.feature import register_translation

image = data.camera()
shift = (100, -50)
offset_image = (0, 0)

offset_image = fourier_shift(np.fft.fftn(image), shift)
offset_image = np.fft.ifftn(offset_image)

print("Poznati pomeraj (y, x): {}".format(shift))

shift_detected, error, diffphase = register_translation(offset_image, image)

fig = plt.figure(figsize=(8, 3))

ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Referentna slika')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Modifikovana slika')

image_product = np.fft.fft2(offset_image) * np.fft.fft2(image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Kroskorelacija")

print("Detektovani pomraj (y, x): {}".format(shift_detected))

plt.show()
