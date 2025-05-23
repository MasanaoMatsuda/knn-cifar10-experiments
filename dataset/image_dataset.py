import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA
from typing import Optional


@dataclass
class ImageDataset:

    def __init__(self, images, labels, name: str, condition: float,
                 reducer: Optional[PCA]=None, do_shuffle: bool=False):
        self.images = images
        self.labels = labels
        self.name: str = f"{name}-{condition}"
        self.condition: float = condition
        self.reducer: Optional[PCA] = reducer
        if do_shuffle:
            self.__shuffle()
    
    @property
    def flat_images(self):
        return self.images.reshape((self.dsize, -1))
    
    @property
    def dsize(self):
        return self.images.shape[0]

    def __shuffle(self):
        idx = np.random.permutation(len(self.labels))
        self.images = self.images[idx]
        self.labels = self.labels[idx]
        print(f"{self.name}データセットをシャッフルしました")

    def set_reducer(self, model: PCA):
        self.reducer = model

    