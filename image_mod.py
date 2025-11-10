from PIL import Image 
import numpy as np
import math

def convole(img, kernel, stride, padding, convolution_func):
    # Basic information gathering
    h, w = img.shape
    kernel_size = len(kernel)
    print(img.shape)
    # Set the output size
    output = np.array([])
    output_h = (h + 2 * padding - kernel_size) // stride + 1
    output_w = (w + 2 * padding - kernel_size) // stride + 1
    output = np.zeros((output_h, output_w))

    # Convolution
    for i in range(0, h - (kernel_size - 1), stride):
        for j in range(0, w - (kernel_size - 1), stride):
            window = img[i:i+kernel_size, j:j+kernel_size]
            new_pixel = convolution_func(window, kernel)
            output[i // stride, j // stride] = new_pixel
    
    return output

def convolution_calc(window, kernel):
    new_pixel = np.sum(window * kernel)
    if new_pixel < 0:
        new_pixel = 0
    return new_pixel

def max_pooling_calc(window, kernel):
    new_pixel = np.max(window)
    return new_pixel

img = Image.open("pixel_mod_original.jpeg")
img = img.convert('L')
pixels = np.array(img)
kernel = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
channels = []

output = convole(pixels, kernel, 1, 0, convolution_calc)
channels.append(output)

kernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
output = convole(pixels, kernel, 1, 0, convolution_calc)
channels.append(output)

pooling_channels = []
# POOLING LAYER
for channel in channels:
    kernel = np.array([[0, 0],
                       [0, 0]])
    output = convole(channel, kernel, 2, 0, max_pooling_calc)
    pooling_channels.append(output)

print(output.shape)
blurred_img = Image.fromarray(output).convert('RGB')
blurred_img.save("blurred_image.jpg")

