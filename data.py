import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, image_dir, csv_path):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = {}
        image_name = self.df['Image_Name'][index]
        image = cv2.cvtColor(cv2.imread(f"{self.image_dir}/{image_name}"), cv2.COLOR_BGR2RGB)
        data['image'] = image
        height, width = image.shape[0], image.shape[1]
        tmp_list = []

        columns = ['Fish', 'Flower', 'Gravel', 'Sugar']
        bg_mask = np.ones((height,width), dtype=np.uint8)
        for column in columns:
            if pd.isnull(self.df[column][index]):
                tmp_list.append(np.zeros((height,width), dtype=np.uint8))
            else:
                tmp_list.append(decode_rle(self.df[column][index],height,width, bg_mask))

        tmp_list.append(bg_mask)
        data['mask'] = np.stack(tmp_list, axis = 2)
        return data

def decode_rle(mask_rle,height, width, bg_mask):
    """decoding mask string"""
    s = mask_rle.split()

    starts , lengths = [np.asarray(x, dtype = int) for x in (s[0::2],s[1::2])]
    starts -= 1

    ends = starts + lengths
    tmp_img = np.zeros(height*width, dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        tmp_img[lo:hi] = 1
    
    image_mask = tmp_img.reshape((height, width), order ="F")
    indices_to_replace = np.where(image_mask == 1)
    bg_mask[indices_to_replace] = 0

    return image_mask