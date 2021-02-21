'''
Tests for Functions in read_idx.py
'''
import sys
sys.path.append('../')

import pytest
import read_idx as rd
import numpy as np
import tempfile
import subprocess


@pytest.fixture(scope="module",autouse=True)
def print_new_line():
    print("")

'''
=================
Helper Functions
=================
'''

def gunzip_to_dir(ip_file_name,ip_file_path,op_file_path):
    # Copy GZ to Temp Directory
    op = subprocess.run(['cp',''.join([ip_file_path,ip_file_name]),''.join([op_file_path,'/'])])
    if op.returncode:
        pytest.fail("Unable to Copy {} to temp directory {}".format(ip_file_name,op_file_path))

    # Extract File in Temp Directory
    op = subprocess.run(['gunzip',''.join([op_file_path,'/',ip_file_name])],stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    if op.returncode:
        pytest.fail("Received errorcode {} when trying to uncompress {}".format(op.returncode,ip_file_path))

    path_to_extracted_file = ''.join([op_file_path,'/',ip_file_name[:-3]])

    return path_to_extracted_file

'''
============
Begin Tests
============
'''

@pytest.mark.parametrize(
        "file_name,file_path,true_metadata",
        [("train-images-idx3-ubyte.gz","../../data/mnist/",('B',3,16,(60000,28,28))),
         ("train-labels-idx1-ubyte.gz","../../data/mnist/",('B',1,8,(60000,))),
         ("t10k-images-idx3-ubyte.gz","../../data/mnist/",('B',3,16,(10000,28,28))),
         ("t10k-labels-idx1-ubyte.gz","../../data/mnist/",('B',1,8,(10000,)))]
        )
def test_magic_number_parsing(file_name,file_path,true_metadata):

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        metadata = rd.get_metadata(file_to_read_full_path)

    assert metadata == true_metadata

# TODO: Better (More Generic) Test for get_data()
def test_get_data_lecun_idx_matrix(debug=False):

    file_name = "train-images-idx3-ubyte.gz"
    file_path = "../../data/mnist/"

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        metadata = rd.get_metadata(file_to_read_full_path)
        # Run get_data()
        idx_data = rd.get_data(file_to_read_full_path,metadata)

    # Set Debug = True to View the First 3 Images : (5,0,4)
    if debug:
        with np.printoptions(threshold=np.inf,linewidth=150):
            print(idx_data[0:3,:,:])

    assert idx_data.shape == (60000,28,28)

# TODO: Better (More Generic) Test for get_data()
def test_get_data_lecun_idx_vector(debug=False):

    file_name = "train-labels-idx1-ubyte.gz"
    file_path = "../../data/mnist/"

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        metadata = rd.get_metadata(file_to_read_full_path)
        # Run get_data()
        idx_data = rd.get_data(file_to_read_full_path,metadata)

    if debug:
        print(idx_data[0:3])

    assert idx_data.shape == (60000,)

    # The following assert is not suitable for floating point or for NaN
    # Just a stop-gap for now
    assert (idx_data[0:9] == np.array([5,0,4,1,9,2,1,3,1])).all()
