'''
Tests for Functions in read_idx.py
'''
import sys
sys.path.append('../')

import read_idx as rd
import tempfile
import subprocess

def test_load_header_lecun_mnist():
    with tempfile.TemporaryDirectory() as dirpath:
        file_path = "../../data/mnist/train-images-idx3-ubyte.gz"
        extracted_file_path = ''.join([dirpath,'/','op_file'])
        
        # Extract GZ to Temp Directory
        op = subprocess.run(['zcat',file_path,'>',extracted_file_path],stderr=subprocess.PIPE,stdout=subprocess.PIPE,shell=True)
        print(op.stderr.decode())
        assert not op.returncode, "Received errorcode {} when trying to uncompress {}".format(op.returncode,file_path)

        subprocess.run(['ls','-ltr',dirpath])
