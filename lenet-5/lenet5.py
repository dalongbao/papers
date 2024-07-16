import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
import os

"""
Structure:
    * Image (28, 28, 1)
    * Conv: (28, 28, 6), kernel_size=5, padding=2
    * Sigmoid
    * Pool: (14, 14, 6), kernel_size=2, stride=2
    * Conv: (10, 10, 16), kernel_size=5, no padding
    * Sigmoid
    * Pool: (5, 5, 16), kernel_size=2, stride=2
    * Flatten
    * Dense: 120 FC neurons
    * Sigmoid
    * Dense: 84 FC neurons
    * Sigmoid
    * Dense: 10 FC neurons
    * Output (10, 1)
"""

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.dense1 = nn.Linear(input_dims=120, output_dims=84)
        self.dense2 = nn.Linear(input_dims=84, output_dims=10)

        self.tanh = nn.Tanh()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.tanh(x)

        x = mx.squeeze(x, (1, 2))

        x = self.dense1(x)
        x = self.tanh(x)
        x = self.dense2(x)
        
        return x

def loss_fn(model, X, y):
    logits = model(X)
    return mx.mean(nn.losses.cross_entropy(logits, y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for i in range(0, y.size, batch_size):
        ids = perm[i: i + batch_size]
        yield X[ids], y[ids]

def predict(model, image):
    image = mx.array(image)
    image = image.reshape(-1, 1)
    prediction = model(image)
    predicted_class = mx.argmax(prediction, axis=1)
    return predicted_class.item()

def load_data():
    BASE_PATH = os.path.join(os.getcwd(), 'mnist/')

    train = os.path.join(BASE_PATH, 'train.csv')
    test = os.path.join(BASE_PATH, 'test.csv')

    train = np.transpose(np.array(pd.read_csv(train)))
    n, m = train.shape
    train_images = np.reshape(np.transpose(train[1:n] / 255.), (-1, 28, 28))
    train_labels = train[0]

    test = np.transpose(np.array(pd.read_csv(test)))
    i, h = test.shape
    test_images = np.reshape(np.transpose(test[1:n] / 255.), (-1, 28, 28))
    test_labels = test[0]

    train_images = mx.expand_dims(mx.array(train_images), -1)
    train_labels = mx.array(train_labels)
    test_images = mx.expand_dims(mx.array(test_images), -1)
    test_labels = mx.array(test_labels)

    return train_images, train_labels, test_images, test_labels

def main():
    model = LeNet()
    mx.eval(model.parameters())

    lr = 1e-4
    num_epochs = 100

    train_images, train_labels, test_images, test_labels = load_data()

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=lr, momentum=0.9)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(32, train_images, train_labels):
            loss, grad = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)

        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()

        print(f'Epoch {e} | Accuracy: {accuracy.item():.3f} | Time: {(toc - tic):.3f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a simple MLP on MNIST')
    parser.add_argument('--gpu', action='store_true', help='Use the Metal backend')
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main()


