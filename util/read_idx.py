'''
Contains Functions to Read IDX Files
'''

'''
Loops Through The File and Collects Header Values
Assumes Every Header is 32-bits Long
Returns List of Header Values in Order -> [H1,H2,H3,...]
'''
def get_headers(input_file_path,offset_data_start,is_big_endian=True):
    if is_big_endian:
        byteorder_string = 'big'
    else:
        byteorder_string = 'little'

    header_list = list()

    # Open File
    ip_file = open(input_file_path,'rb')

    i = 0
    while(offset_data_start):
        val = int.from_bytes(ip_file.read(4), byteorder=byteorder_string)
        header_list.append(val)
        offset_data_start-=4
        i+=1
    
    return header_list
