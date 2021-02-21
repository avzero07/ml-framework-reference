'''
Contains Functions to Read IDX Files
'''

import numpy as np

'''
==================================================
Private Helper Functions to Reduce Redundant Code
==================================================
'''

# Get Byte-Order String
def _get_byteorder_string(is_big_endian):
    if is_big_endian:
        return 'big'
    else:
        return 'little'

# Interpret Datatype
def _get_datatype(dtype_number):
    dtype_dict = {
            0x08: 'B',  # unsigned byte
            0x09: 'b',  # signed byte
            0x0B: 'h',  # short (2 Bytes)
            0x0C: 'i',  # int (4 Bytes)
            0x0D: 'f',  # float (4 Bytes)
            0x0E: 'd'   # double (8 Bytes)
            }
    return dtype_dict[dtype_number]

'''
====================================
Functions to Read/Process IDX Files
====================================
'''

'''
Reads and interprets the Magic number in the
datastore.

Returns (data_type, dim_count, data_offset)
- data_type     :   data type (of data)
- dim_count     :   number of dimensions
- data_offset   :   data start offset

'''
def get_metadata(input_file_path,is_big_endian=True):
    
    byteorder_string = _get_byteorder_string(is_big_endian)
    
    # Open File
    with open(input_file_path,'rb') as ip_file:
        # Assume Magic Number is Always 32-bit
        data_offset = 0
        # Discard Byte 1 and Byte 2
        ip_file.read(2)
        data_offset+=2

        # Process 3rd Byte to Get d_type
        data_type = _get_datatype(int.from_bytes(ip_file.read(1), byteorder=byteorder_string))
        data_offset+=1

        # Process 4th Byte to Get Number of Dimensions
        dim_count = int.from_bytes(ip_file.read(1), byteorder=byteorder_string)
        data_offset+=1
        
        # Get Shape
        shape = ()
        for i in range(dim_count):
            num_elem_in_dim = int.from_bytes(ip_file.read(4), byteorder=byteorder_string)
            data_offset+=4
            shape = shape+(num_elem_in_dim,)

    return (data_type, dim_count, data_offset, shape)

'''
Loops Through IDX File and Returns Numpy Array of Data
'''
def get_data(input_file_path,metadata,is_big_endian=True):

    byteorder_string = _get_byteorder_string(is_big_endian)

    data_offset = metadata[2]
    data_shape = metadata[3]

    # TODO: Make this better when generalizing
    if len(data_shape)==1:
        target_shape = (data_shape[0],1,1)
    elif len(data_shape)>1:
        target_shape = data_shape

    # Open File
    with open(input_file_path,'rb') as ip_file:
        headers = ip_file.read(data_offset)
        op = np.zeros(target_shape)
        for n in range(target_shape[0]):
            for r in range(target_shape[1]):
                for c in range(target_shape[2]):
                    op[n,r,c] = int.from_bytes(ip_file.read(1), byteorder=byteorder_string)

    return op.squeeze()
