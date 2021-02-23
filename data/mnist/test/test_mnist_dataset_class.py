'''
Tests for mnist_dataset Class
'''
import os
import sys
import tempfile
sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..","..","..","util"))
sys.path.append(os.path.join("..","..","..","util","test"))

from mnist_dataset import MNISTDataset, MismatchedDataError
from read_idx import get_data, get_metadata
from test_read_idx import gunzip_to_dir

import pytest

@pytest.mark.parametrize(
        "image_file_name,label_file_name,file_path,outcome",
        [("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",os.path.join(".."),True),
         ("t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",os.path.join(".."),True),
         ("t10k-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",os.path.join(".."),False),
         ("train-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",os.path.join(".."),False)
            ])
def test_mnist_dataset_init(image_file_name,label_file_name,file_path,outcome):
    with tempfile.TemporaryDirectory() as dirpath:
        image_file_to_read_full_path = gunzip_to_dir(image_file_name,file_path,dirpath)
        label_file_to_read_full_path = gunzip_to_dir(label_file_name,file_path,dirpath)

        try:
            dataset = MNISTDataset(image_file_to_read_full_path,label_file_to_read_full_path)
            res = True

        except MismatchedDataError as e:
            res = False

        assert res == outcome, "MNISTDatset class __init__ method is broken!"

@pytest.mark.parametrize(
        "image_file_name,label_file_name,file_path,true_length",
        [("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",os.path.join(".."),60000),
         ("t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",os.path.join(".."),10000)
            ])
def test_mnist_dataset_len(image_file_name,label_file_name,file_path,true_length):
    with tempfile.TemporaryDirectory() as dirpath:
        image_file_to_read_full_path = gunzip_to_dir(image_file_name,file_path,dirpath)
        label_file_to_read_full_path = gunzip_to_dir(label_file_name,file_path,dirpath)

        dataset = MNISTDataset(image_file_to_read_full_path,label_file_to_read_full_path)

        assert len(dataset) == true_length, "MNISTDatset class __len__ method is broken!"

# TODO: Simple test for now. Will need to re-work
def test_mnist_dataset_get_item():
    image_file_name = "train-images-idx3-ubyte.gz"
    label_file_name = "train-labels-idx1-ubyte.gz"
    file_path = os.path.join("..")

    with tempfile.TemporaryDirectory() as dirpath:
        image_file_to_read_full_path = gunzip_to_dir(image_file_name,file_path,dirpath)
        label_file_to_read_full_path = gunzip_to_dir(label_file_name,file_path,dirpath)

        dataset = MNISTDataset(image_file_to_read_full_path,label_file_to_read_full_path)
        
        message = "MNISTDataset class __getitem__ method is broken!"
        
        sample_1 = dataset[0]
        assert sample_1['label'] == 5, message

        sample_2 = dataset[1]
        assert sample_2['label'] == 0, message

        sample_3 = dataset[2]
        assert sample_3['label'] == 4, message
