'''
Defines a Custom Class for MNIST that
derives from torch.utils.data.Dataset
'''

import os
import sys
sys.path.append(os.path.join("..","..","util"))
import read_idx as rd

import torch

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self,path_to_image_store,path_to_labels,transform=None):
        # Load the Data
        im_metadata = rd.get_metadata(path_to_image_store)
        lb_metadata = rd.get_metadata(path_to_labels)

        self.image_data = rd.get_data(path_to_image_store,im_metadata)
        self.label_data = rd.get_data(path_to_labels,lb_metadata)
        
        # Convert to Tensors
        self.image_data = torch.tensor(self.image_data,dtype=torch.float)
        self.label_data = torch.tensor(self.label_data,dtype=torch.long)

        # Fix Dimensions of Images
        self.image_data = self.image_data.unsqueeze(1)

        if self.image_data.size(0) != self.label_data.size(0):
            raise MismatchedDataError("len(Images) != len(labels). Check loaded data!")

        self.transform = transform

    def __len__(self):
        return self.label_data.size(0)

    def __getitem__(self,index):
        image = self.image_data[index,]
        label = self.label_data[index,]

        sample = {'image':image,'label':label}

        # TODO: Figure out what transforms make sense
        if self.transform:
            print("Transforms are Unsupported at this time!")
    
        return sample


class MismatchedDataError(Exception):
    def __init__(self,message="Something is Wrong With Loaded Data"):
        self.message = message
        super().__init__(self.message)
