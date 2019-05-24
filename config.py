from abc import ABC
from collections import defaultdict


class AbstractConfig(ABC):

    @property
    def checkpoint_path(self):
        raise NotImplementedError("You should defined 'checkpoint_path'.")

    @property
    def mean(self):
        raise NotImplementedError("You should defined 'mean'.")

    @property
    def std(self):
        raise NotImplementedError("You should defined 'std'.")

    @property
    def resize_to(self):
        raise NotImplementedError("You should defined 'resize_to'.")

    crop_to = None


class Default(AbstractConfig):
    mean = ( 0.485, 0.456, 0.406 )
    std = ( 0.229, 0.224, 0.225 )
    resize_to = ( 224, 224 )

class Inception(AbstractConfig):
    mean = ( 0.5, 0.5, 0.5 )
    std = ( 0.5, 0.5, 0.5 )
    resize_to = ( 299, 299 )


class C3D(AbstractConfig):
    checkpoint_path = "pretrained_models/c3d.pickle"
    import numpy as np
    mean = np.load("data/c3d_mean.npy")
    mean = mean.squeeze(0).transpose(1, 2, 3, 0)
    mean /= 255.
    std = ( 0.5, 0.5, 0.5 )
    resize_to = ( 171, 128 )
    crop_to = ( 112, 112 )



config = defaultdict(lambda: Default)
config['inception_v3'] = Inception
config['inception_v4'] = Inception
config['c3d'] = C3D

