import gzip
import os
import numpy as np
from urllib import request

def download_mnist():
    base = "http://yann.lecun.com/exdb/mnist/"
    files = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
            ]

    for file in files
