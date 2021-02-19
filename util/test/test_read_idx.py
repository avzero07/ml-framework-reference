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
Helper Functions
'''

def gunzip_to_dir(ip_file_name,ip_file_path,op_file_path):
    # Copy GZ to Temp Directory
    op = subprocess.run(['cp',''.join([ip_file_path,ip_file_name]),''.join([op_file_path,'/'])])
    assert not op.returncode, "Unable to Copy {} to temp directory {}".format(ip_file_name,op_file_path)

    # Extract File in Temp Directory
    op = subprocess.run(['gunzip',''.join([op_file_path,'/',ip_file_name])],stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    assert not op.returncode, "Received errorcode {} when trying to uncompress {}".format(op.returncode,ip_file_path)

    path_to_extracted_file = ''.join([op_file_path,'/',ip_file_name[:-3]])

    return path_to_extracted_file

def test_get_header_lecun_idx():

    file_name = "train-images-idx3-ubyte.gz"
    file_path = "../../data/mnist/"

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        # Run get_headers()
        header_list = rd.get_headers(file_to_read_full_path,16)

    '''
    LeCun's IDX Format Has The Following Structure For 'train-images-idx3-ubyte'

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    '''

    true_header_list = [2051,60000,28,28]

    for i in range(len(header_list)):
        assert true_header_list[i] == header_list[i], "Header{} Mismatch!".format(i)

# TODO: Better (More Generic) Test for get_data()
def test_get_data_lecun_idx_matrix():

    file_name = "train-images-idx3-ubyte.gz"
    file_path = "../../data/mnist/"
    data_shape = (60000,28,28)

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        # Run get_data()
        idx_data = rd.get_data(file_to_read_full_path,16,data_shape)

    # Uncomment Next Two Lines To View First 3 Images (5,0,4)

    #with np.printoptions(threshold=np.inf,linewidth=150):
    #    print(idx_data[0:3,:,:])

    assert idx_data.shape == (60000,28,28)

# TODO: Better (More Generic) Test for get_data()
def test_get_data_lecun_idx_vector():

    file_name = "train-labels-idx1-ubyte.gz"
    file_path = "../../data/mnist/"
    data_shape = (60000,1,1)

    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(file_name,file_path,dirpath)
        # Run get_data()
        idx_data = rd.get_data(file_to_read_full_path,8,data_shape)

    #print(idx_data[0:3])

    assert idx_data.shape == (60000,)

    # The following assert is not suitable for floating point or for NaN
    # Just a stop-gap for now
    assert (idx_data[0:9] == np.array([5,0,4,1,9,2,1,3,1])).all()
