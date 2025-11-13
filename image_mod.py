from PIL import Image 
import numpy as np
from MNIST_data_processing import load_mnist_images, load_mnist_labels, build_dataset

# def convole(img, kernel, stride, padding, convolution_func):
#     # Basic information gathering
#     h, w = img.shape
#     kernel_size = len(kernel)
#     # Set the output size
#     output = np.array([])
#     output_h = (h + 2 * padding - kernel_size) // stride + 1
#     output_w = (w + 2 * padding - kernel_size) // stride + 1
#     output = np.zeros((output_h, output_w))

#     # Convolution
#     for i in range(0, h - (kernel_size - 1), stride):
#         for j in range(0, w - (kernel_size - 1), stride):
#             window = img[i:i+kernel_size, j:j+kernel_size]
#             new_pixel = convolution_func(window, kernel)
#             output[i // stride, j // stride] = new_pixel
    
#     return output

# def convolution_calc(window, kernel):
#     new_pixel = np.sum(window * kernel)
#     new_pixel = max(0, new_pixel)
#     return new_pixel

# def max_pooling_calc(window, kernel):
#     new_pixel = np.max(window)
#     return new_pixel

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilizing to prevent overflow
    return exp_x / np.sum(exp_x)

def softmax_prime(s):
    # s is the softmax output: shape (n,1) or (n,)
    s = s.reshape(-1, 1)        # ensure column vector
    return np.diagflat(s) - s @ s.T


# img = images[0]
# pixels = np.array(img)
# kernels = [ np.array([[-1, -2, -1],
#                     [0, 0, 0],
#                     [1, 2, 1]]),
#             np.array([[-1, 0, 1],
#                     [-2, 0, 2],
#                     [-1, 0, 1]])]
# channels = []

# # FIRST CONVOLUTION LAYER
# output = convole(pixels, kernels[0], 1, 0, convolution_calc)
# channels.append(output)

# output = convole(pixels, kernels[1], 1, 0, convolution_calc)
# channels.append(output)

# pooling_channels = []
# # FIRST POOLING LAYER
# for channel in channels:
#     kernel = np.array([[0, 0],
#                        [0, 0]])
#     output = convole(channel, kernel, 2, 0, max_pooling_calc)
#     pooling_channels.append(output)

# # SECOND CONVOLUTION LAYER
# channels = []
# for i, pooling_channel in enumerate(pooling_channels):
#     output = convole(pooling_channel, kernels[i], 1, 0, convolution_calc)
#     channels.append(output)

# pooling_channels = []
# # SECOND POOLING LAYER
# for channel in channels:
#     kernel = np.array([[0, 0],
#                        [0, 0]])
#     output = convole(channel, kernel, 2, 0, max_pooling_calc)
#     pooling_channels.append(output)  

# # FLATTEN THE ARRAY
# flattened_array = np.vstack(pooling_channels).ravel()

# fully_connected_weights = []
# biases = []
# activations = []

# # FIRST LAYER IN THE FULLY CONNECTED LAYER
# bias = 0
# biases.append(bias)

# w1 = np.full((flattened_array.size, 16), 0.01)
# fully_connected_weights.append(w1)

# a1 = np.dot(flattened_array, w1)
# activations.append(a1)

# # SECOND LAYER IN THE FULLY CONNECTED LAYER
# bias = 0
# biases.append(bias)

# w2 = np.full((a1.size, 16), 0.01)
# fully_connected_weights.append(w2)

# a2 = np.dot(a1, w2)
# activations.append(a2)

# # FULLY CONNECTED LAYER
# bias = 0
# biases.append(bias)

# w3 = np.full((a2.size, 10), 0.01)
# fully_connected_weights.append(w3)

# res = np.dot(a2, w3)
# activations.append(res)

# softmax_res = softmax(res)
# print(softmax_res)

# # COST CALCULATION
# cost = 0
# for i, out in enumerate(softmax_res):
#     if i == labels[0]:
#         cost += (out - 1) ** 2
#     else:
#         cost += (out - 0) ** 2
# print(cost)
# print(activations[2])

# deltas = []
# weight_deltas = []
# # CALCULATE THE DERIVATIVES FOR THE OUTPUT LAYER
# for j, activation in enumerate(activations[2]):
#     relu_derivative = 0
#     if activation < 0:
#         relu_derivative = 0
#     else:
#         relu_derivative = 1
#     delta = 2 * (expected - activation) * activation * relu_derivative
#     deltas.append(delta)

#     for prev_activation in activations[1]:
#         weight_delta = prev_activation * delta
#         weight_deltas.append(weight_delta)
    



# blurred_img = Image.fromarray(output).convert('RGB')
# blurred_img.save("blurred_image.jpg")

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(int)

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.weighted_inputs = [None for _ in sizes[1:]]
        self.activations = [None for _ in sizes]
        self.gradient_weights = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.gradient_biases  = [np.zeros((y, 1)) for y in sizes[1:]]


    def feedforward(self, input):
        # print("NEW FORWARD PASS")
        self.activations[0] = input
        for i in range(1, self.num_layers):
            # print("input: ", input)
            # print("weights: ", self.weights[i - 1])
            # print("biases: ", self.biases[i - 1])
            # print("input shape: ", input.shape)
            # print("weights shape: ", self.weights[i - 1].shape)
            z = np.dot(self.weights[i - 1], input) + self.biases[i - 1]
            # print("output shape: ", z.shape)
            self.weighted_inputs[i - 1] = z
            if i == self.num_layers - 1:
                out = softmax(z)
            else:
                out = relu(z)
            self.activations[i] = out
            input = out
        return out
    
    def calculate_cost(self, output, expected):
        cost = -np.sum(expected * np.log(output + 1e-12))
        return cost  # prevent log(0)

    
    def backpropagation(self, y):
        # ---- OUTPUT LAYER ----
        delta = self.activations[-1] - y   # EASY & CORRECT
        
        self.gradient_biases[-1] = delta
        self.gradient_weights[-1] = delta @ self.activations[-2].T

        # ---- HIDDEN LAYERS ----
        for l in range(2, self.num_layers):
            z = self.weighted_inputs[-l]
            rp = relu_prime(z)

            delta = (self.weights[-l+1].T @ delta) * rp

            self.gradient_biases[-l] = delta
            self.gradient_weights[-l] = delta @ self.activations[-l-1].T
        return 

    def update_weights(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.gradient_weights[i]
            self.biases[i]  -= lr * self.gradient_biases[i]

    def train(self, data, lr, epochs):
        for epoch in range(epochs):
            np.random.shuffle(data)
            total_cost = 0
            
            for x, y in data:
                out = self.feedforward(x)
                total_cost += self.calculate_cost(out, y)

                self.backpropagation(y)
                self.update_weights(lr)

            print(f"Epoch {epoch+1}, cost = {total_cost}")

    def predict(self, input):
        output = self.feedforward(input)
        print(output)
        prediction = np.argmax(output)
        return prediction


# net = Network([2, 3, 2])
# input = np.array([[0.7], [0.2]])
# output = softmax(net.feedforward(input))
# expected = np.array([[1], [0]])
# cost = net.calculate_cost(output, expected)

# Load raw MNIST
train_images = load_mnist_images("train-images.idx3-ubyte")
train_labels = load_mnist_labels("train-labels.idx1-ubyte")

# Build dataset (ready for neural net)
train_data = build_dataset(train_images, train_labels)

x, y = train_data[0]
print(x.shape, y.shape)

net = Network([784, 16, 16, 10])
net.train(train_data, 0.001, 25)

# Load raw MNIST
test_images = load_mnist_images("t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

correct = 0
for i in range(100):
    prediction = net.predict(np.array(test_images[i]).flatten().reshape(784, 1))
    answer = test_labels[i]
    if prediction == answer:
        correct += 1
    else:
        print("predcition: ", prediction)
        print("answer: ", answer)
print("accuracy: ", correct/100)

