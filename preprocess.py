from torch.utils.data import Dataset
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread

def get_image_data(root_dir):
    image_name_list = os.listdir(root_dir)
    image_list  = np.zeros((len(image_name_list), 64, 64, 3))
    for i in tqdm(range(len(image_name_list))):
        filename = image_name_list[i]
        path = os.path.join(root_dir, filename)
        image = imread(path)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        image = resize(image, (64, 64, 3))
        image_list[i] = image

    # pytorch Conv2D expects num_channels first 
    image_list = np.swapaxes(image_list, -1, 1)
    return image_list

class SourceDataset(Dataset):
  def __init__(self, root_dir):
    self.source_imgs = get_image_data(root_dir)
      
  def __len__(self):
    return len(self.source_imgs)

  def __getitem__(self, idx):
    item = {"source_img": self.source_imgs[idx]}
    return item 

class TargetDataset(Dataset):
  def __init__(self, root_dir):
    self.target_imgs = get_image_data(root_dir)
      
  def __len__(self):
    return len(self.target_imgs)

  def __getitem__(self, idx):
    item = {"target_img": self.target_imgs[idx]}
    return item 

