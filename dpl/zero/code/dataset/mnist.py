import urllib.request
import os.path
import zipfile
import pickle
import os
import numpy as np


url_base = "https://yann.lecun.com/exdb/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

dataset_dir = os.path.dirname(os.path / avspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("downloading " + file_name + " ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
    }
    request = urllib.request.Request(url_base + file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode="wb") as f:
        f.write(response)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)

def _load_label(file_name):
    file_path = dataset_dir + '/' + file_name

    print('Converting ' + file_name + ' to Numpy Array ...')
    with gzip.open
