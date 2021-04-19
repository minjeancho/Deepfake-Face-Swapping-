from torch.utils.data import Dataset
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread

def get_image_data(root_dir):

    #get image data reference on HW4 preprocess.py and run.py
    image_name_list = os.listdir(root_dir)

    # Get image data, resize to 64X64X3
    image_list  = np.zeros((len(image_name_list), 64, 64, 3))

    for i in range(len(image_name_list)):
        filename = image_name_list[i]
        path = os.path.join(root_dir, filename)
        image = imread(path)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        image = resize(image, (64, 64, 3))
        image_list[i] = image
    
    print(np.shape(image_list))
    return image_list

#TODO preprocess data 
class DeepfakeDataset(Dataset):
    def __init__(self,root_dir):
        #output image should be of size 64x64x3 (height x width x num_channels)

        #Please give the dir for incoming data because train and test data are usually in different directories
        #eg: for train, the root_dir should be /xxxx/train
        self.root_dir = root_dir

        #get image data reference on HW4 preprocess.py and run.py
        image_name_list = os.listdir(self.root_dir)

        # Get image data, resize to 64X64X3
        image_list  = np.zeros(
            (len(image_name_list), 64, 64, 3))
        label_list = image_name_list

        for i in range(len(image_name_list)):
            filename = image_name_list[i]
            path = os.path.join(self.root_dir, filename)
            image = imread(path)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            image = resize(image, (64, 64, 3))
            image_list[i] = image

        self.image_list = image_list
        # Currently set label = image_name because I do not know what will be the actual image name
        # This will be  modified after determined the data we use
        self.label_list = label_list

        # I assume you do not pass transform information, so I did not set transform attribute here
        #self.transform = transform
        
    def __len__(self):

        return len(self.image_list)
        
    
    def __getitem__(self, idx):
        img = self.image_list[idx]
        label = self.label_list[idx]

        sample = {'image':img,'label':label}

        return sample


    