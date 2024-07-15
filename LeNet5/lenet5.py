import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import time

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
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = mx.flatten()

        self.dense1 = nn.Linear(input_dims=400, output_dims=120)
        self.dense2 = nn.Linear(input_dims=120, output_dims=84)
        self.dense3 = nn.Linear(input_dims=84, output_dims=10)

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = self.dense3(x)
        
        return x

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

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

def main():
    model = LeNet()
    mx.eval(model.parameters())

    lr = 1e-1
    num_epochs = 10

    # Load data here as train_images, train_data, val_images, val_data

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr)

    for i in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(256, train_images, train_data):
            loss, grad = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
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


