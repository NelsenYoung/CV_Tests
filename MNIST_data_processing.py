import numpy as np
import struct

# --- Read MNIST image file ---
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read header: magic number, number of images, rows, cols
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read remaining bytes as unsigned 8-bit integers
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape into 3D array [num_images, rows, cols]
        images = data.reshape(num_images, rows, cols)
        return images

# --- Read MNIST label file ---
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Example usage:
images = load_mnist_images("t10k-images.idx3-ubyte")
labels = load_mnist_labels("t10k-labels.idx1-ubyte")

# print("Images shape:", images.shape)   # (10000, 28, 28)
# print("Labels shape:", labels.shape)   # (10000,)
# print("First label:", labels[0])
# print("First image:\n", images[0])

# Normalize for CNN input
images = images.astype(np.float32) / 255.0
