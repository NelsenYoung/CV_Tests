from PIL import Image 
import numpy as np
from MNIST_data_processing import images, labels

def convole(img, kernel, stride, padding, convolution_func):
    # Basic information gathering
    h, w = img.shape
    kernel_size = len(kernel)
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
    new_pixel = max(0, new_pixel)
    return new_pixel

def max_pooling_calc(window, kernel):
    new_pixel = np.max(window)
    return new_pixel

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilizing to prevent overflow
    return exp_x / np.sum(exp_x)

img = images[0]
pixels = np.array(img)
kernels = [ np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]),
            np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])]
channels = []

# FIRST CONVOLUTION LAYER
output = convole(pixels, kernels[0], 1, 0, convolution_calc)
channels.append(output)

output = convole(pixels, kernels[1], 1, 0, convolution_calc)
channels.append(output)

pooling_channels = []
# FIRST POOLING LAYER
for channel in channels:
    kernel = np.array([[0, 0],
                       [0, 0]])
    output = convole(channel, kernel, 2, 0, max_pooling_calc)
    pooling_channels.append(output)

# SECOND CONVOLUTION LAYER
channels = []
for i, pooling_channel in enumerate(pooling_channels):
    output = convole(pooling_channel, kernels[i], 1, 0, convolution_calc)
    channels.append(output)

pooling_channels = []
# SECOND POOLING LAYER
for channel in channels:
    kernel = np.array([[0, 0],
                       [0, 0]])
    output = convole(channel, kernel, 2, 0, max_pooling_calc)
    pooling_channels.append(output)  

# FLATTEN THE ARRAY
flattened_array = np.vstack(pooling_channels).ravel()

# FULLY CONNECTED LAYER
bias = 0
arr = np.full((flattened_array.size, 10), 0.01)
res = np.dot(flattened_array, arr)
softmax_res = softmax(res)
print(softmax_res)

# COST CALCULATION
cost = 0
for i, out in enumerate(softmax_res):
    if i == labels[0]:
        cost += (out - 1) ** 2
    else:
        cost += (out - 0) ** 2
print(cost)

# blurred_img = Image.fromarray(output).convert('RGB')
# blurred_img.save("blurred_image.jpg")

