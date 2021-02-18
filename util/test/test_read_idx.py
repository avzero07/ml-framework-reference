'''
Tests for Functions in read_idx.py
'''
import sys
sys.path.append('../')

import pytest
import read_idx as rd
import tempfile
import subprocess

@pytest.fixture(scope="module",autouse=True)
def print_new_line():
    print("")

def test_get_header_lecun_mnist():
    with tempfile.TemporaryDirectory() as dirpath:
        file_name = "train-images-idx3-ubyte.gz"
        file_path = "../../data/mnist/"
        
        # Copy GZ to Temp Directory
        op = subprocess.run(['cp',''.join([file_path,file_name]),''.join([dirpath,'/'])])
        assert not op.returncode, "Unable to Copy {} to temp directory {}".format(file_name,dirpath)
        
        # Extract File in Temp Directory
        op = subprocess.run(['gunzip',''.join([dirpath,'/',file_name])],stderr=subprocess.PIPE,stdout=subprocess.PIPE)
        assert not op.returncode, "Received errorcode {} when trying to uncompress {}".format(op.returncode,file_path)

        # Run get_headers()
        header_list = rd.get_headers(''.join([dirpath,'/',file_name[:-3]]),16)

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
