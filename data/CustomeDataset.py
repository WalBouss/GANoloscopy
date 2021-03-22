from torch.utils.data import Dataset

import os
from PIL import Image

from data.generate_mask import init_mask


class MaskDataset(Dataset):
    '''
        Create a dataset class that output a batch of masks
        output: masks = batch
        '''
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        self.list_file_name = os.listdir(self.rootdir)

    def __len__(self):
        return len([name for name in os.listdir(self.rootdir)])# if os.path.isfile(name)])

    def __getitem__(self, item):
        msk_path = os.path.join(self.rootdir, self.list_file_name[item])
        mask = Image.open(msk_path)
        if self.transform:
            mask = self.transform(mask)

        return mask


class PolypNMaskDataset(Dataset):
    '''
    Create a dataset class that output a batch of images,mask
    output: images, masks = batch
    '''
    def __init__(self, dir_img, dir_msk, transform=None):
        self.dir_img = dir_img
        self.dir_msk = dir_msk
        self.transform = transform
        self.list_file_name = os.listdir(dir_img)

    def __len__(self):
        return len([name for name in self.list_file_name])# if os.path.isfile(name)])

    def __getitem__(self, item):
        img_path = os.path.join(self.dir_img, self.list_file_name[item])
        msk_path = os.path.join(self.dir_msk, self.list_file_name[item])
        image = Image.open(img_path)
        mask = Image.open(msk_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class GenerateMaskDataset(Dataset):
    def __init__(self, dataset_length, input_size=256, transform=None):
        self.dataset_length = dataset_length
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        mask = init_mask(self.input_size)
        mask = Image.fromarray(mask)
        mask = mask.convert("L")

        if self.transform:
            mask = self.transform(mask)
        return mask
