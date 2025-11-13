import numpy as np
import struct

# ---------------------------------------------------------
# Read MNIST image file
# ---------------------------------------------------------
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)

        return images.astype(np.float32) / 255.0   # normalized to [0,1]


# ---------------------------------------------------------
# Read MNIST label file
# ---------------------------------------------------------
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# ---------------------------------------------------------
# Convert labels to one-hot vectors
# ---------------------------------------------------------
def one_hot(labels, num_classes=10):
    oh = np.zeros((labels.size, num_classes))
    oh[np.arange(labels.size), labels] = 1
    return oh


# ---------------------------------------------------------
# Flatten images for fully connected networks
# (28x28 → 784x1 column vector)
# ---------------------------------------------------------
def flatten_images(images):
    num = images.shape[0]
    return images.reshape(num, -1, 1)   # each image becomes (784, 1)


# ---------------------------------------------------------
# Build dataset ready for feedforward networks
# returns list of (input, label) pairs
# ---------------------------------------------------------
def build_dataset(images, labels, one_hot_labels=True, flatten=True):
    if flatten:
        images = flatten_images(images)

    if one_hot_labels:
        labels = one_hot(labels).reshape(labels.size, -1, 1)  # (10,1)

    dataset = [(images[i], labels[i]) for i in range(len(images))]
    return dataset


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load raw MNIST
    test_images = load_mnist_images("t10k-images.idx3-ubyte")
    test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

    # Build dataset (ready for neural net)
    test_data = build_dataset(test_images, test_labels)

    # Check shapes
    print("Test images shape:", test_images.shape)  # (10000, 28, 28)
    print("First flattened image shape:", test_data[0][0].shape)  # (784, 1)
    print("First one-hot label:", test_data[0][1].T)
