import os
import numpy as np
from torch.utils.data import Dataset

# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

class PRBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        try:
            arr = np.load(path)[np.newaxis] # (1, 128, 128)
        except:
            print(path)    
        arr = np.transpose(arr, (1, 2, 0)) # (128, 128, 1)
        arr = arr.astype(np.float32) / 63.5 - 1
        return arr, path  # save embed as the same name

class PRTrain(PRBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = paths

class PRTest(PRBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = paths
