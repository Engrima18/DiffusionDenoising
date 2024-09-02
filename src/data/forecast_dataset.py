import numpy as np
import importlib
import os

import matplotlib.pyplot as plt
from joblib import dump, load


astro_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class ForecastDataset:
    def __init__(self, folder_images, dim=256):
        module = importlib.import_module(folder_images.replace("/", ".")[:-1])

        # DATA
        self.img = module.img
        self.gth = module.gth

        # STATS
        self.dim = dim
        self.h, self.w = self.img.shape
        
        # PATCHES
        max_h = self.h - (self.h % self.dim)
        max_w = self.w - (self.w % self.dim)
        x = np.arange(0, max_h - self.dim, self.dim)
        y = np.arange(0, max_w - self.dim, self.dim)
        xv, yv = np.meshgrid(x, y)
        
        self.img_xy = np.stack([xv.flatten(), yv.flatten()]).T
        
        self.img_idxs = np.arange(0, self.img_xy.shape[0])


    def __len__(self):
        return len(self.img_idxs)


    def get(self, idx):
        x, y = self.img_xy[self.img_idxs[idx]]
        img = self.img[x:x+self.dim, y:y+self.dim]
        gth = self.gth[x:x+self.dim, y:y+self.dim]
        return {'img':img, 'gth':gth, 'x':x, 'y':y, 'id': self.img_idxs[idx]}
