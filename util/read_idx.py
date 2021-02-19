'''
Contains Functions to Read IDX Files
'''

import numpy as np

'''
Private Helper Functions to Reduce Redundant Code
'''

# Get Byte-Order String
def _get_byteorder_string(is_big_endian):
    if is_big_endian:
        return 'big'
    else:
        return 'little'

'''
Loops Through IDX File and Returns Header Values
Assumes Every Header is 32-bits Long

Returns List of Header Values in Order -> [H1,H2,H3,...]
'''
def get_headers(input_file_path,offset_data_start,is_big_endian=True):

    byteorder_string = _get_byteorder_string(is_big_endian)

    header_list = list()

    # Open File
    with open(input_file_path,'rb') as ip_file:
        i = 0
        while(offset_data_start):
            val = int.from_bytes(ip_file.read(4), byteorder=byteorder_string)
            header_list.append(val)
            offset_data_start-=4
            i+=1

    return header_list

'''
Loops Through IDX File and Returns Numpy Array of Data
Headers are Ignored
Shape Specified Vector/Matrix Shape as a tuple (num_elements,num_rows,num_columns)

Returns Numpy Array Containing Data
'''
def get_data(input_file_path,offset_data_start,shape,is_big_endian=True,is_signed=False):

    byteorder_string = _get_byteorder_string(is_big_endian)

    # Open File
    with open(input_file_path,'rb') as ip_file:
        headers = ip_file.read(offset_data_start)
        op = np.zeros(shape)
        for n in range(shape[0]):
            for r in range(shape[1]):
                for c in range(shape[2]):
                    op[n,r,c] = int.from_bytes(ip_file.read(1), byteorder=byteorder_string)

    return op.squeeze()
