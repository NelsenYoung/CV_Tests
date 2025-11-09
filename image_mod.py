from PIL import Image 
import numpy as np
import math

img = Image.open("pixel_mod_original.jpeg")
img = img.convert('L')

pixels = np.array(img)

kernel = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])
print(kernel.shape)

output = np.array([])

h, w = img.size
blur_rate = 9
index = int(math.sqrt(blur_rate))

output_h = h - index + 1
output_w = w - index + 1
output = np.zeros((output_h, output_w))

for i in range(h):
    cur_row = np.array([])
    for j in range(w):
        window = pixels[i:i+index, j:j+index]
        if len(window) < index or len(window[0]) < index:
            continue
        new_pixel = np.sum(window * kernel)
        new_pixel = abs(int(new_pixel))
        output[i, j] = new_pixel

print(output)
blurred_img = Image.fromarray(output).convert('RGB')
blurred_img.save("blurred_image.jpg")

