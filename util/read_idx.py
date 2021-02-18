'''
Contains Functions to Read IDX Files
'''

'''
Loops Through The File and Prints Out Headers

Assumes Every Header is 32-bits Long
'''
def print_headers(input_file_path,offset_data_starti,is_big_endian=True):
    # Open File
    if is_big_endian:
        byteorder_string = 'big'
    else:
        byteorder_string = 'little'

    ip_file = open(input_file_path,'rb')
    while(offset_data_start):
        val = int.from_bytes(ip_file.read(4), byteorder=byteorder_string)
        print("H1 = {}".format(val))
        offset_data_start-=1
