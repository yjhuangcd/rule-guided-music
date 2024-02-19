import math
import random
import os
import pandas as pd
import re
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

CLUSTERS = {'Balakirev': 0,
            'Bartholdy': 0,
            'Bizet': 0,
            'Brahms': 0,
            'Busoni': 0,
            'Chopin': 0,
            'Grieg': 0,
            'Horowitz': 0,
            'Liszt': 0,
            'Mendelssohn': 0,
            'Moszkowski': 0,
            'Paganini': 0,
            'Saint-Saens': 0,
            'Schubert': 0,
            'Schumann': 0,
            'Strauss': 0,
            'Tchaikovsky': 0,
            'Wagner': 0,
            'Beethoven': 1,
            'Bach': 2,
            'Handel': 2,
            'Purcell': 2,
            'Barber': 3,
            'Bartok': 3,
            'Hindemith': 3,
            'Ligeti': 3,
            'Messiaen': 3,
            'Mussorgsky': 3,
            'Myaskovsky': 3,
            'Prokofiev': 3,
            'Schnittke': 3,
            'Schonberg': 3,
            'Shostakovich': 3,
            'Stravinsky': 3,
            'Debussy': 4,
            'Ravel': 4,
            'Clementi': 5,
            'Haydn': 5,
            'Mozart': 5,
            'Pachelbel': 5,
            'Scarlatti': 5,
            'Rachmaninoff': 6,
            'Scriabin': 6,
            'Gershwin': 7,
            'Kapustin': 7
            }


def extract_string(file_name):
    if 'loc' not in file_name:
        ind = [i.start() for i in re.finditer('_', file_name)][-1]
        name = file_name[:ind]
    else:
        name = file_name.split('loc')[0][:-1]
    return name


def find_composer(name, df):
    compound_composer = df.loc[df['simple_midi_name'] == name]['canonical_composer'].item()
    composer = compound_composer.split(' / ')[0].split(' ')[-1]   # take the last name of the first composer
    result = CLUSTERS.setdefault(composer, 8)    # default cluster is everyone else (8)
    return result


def load_data(
    *,
    data_dir,
    batch_size,
    class_cond=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files(data_dir)
    classes = None
    if class_cond:
        # find the composer
        parent_dir = os.path.join(*data_dir.split('/')[:-1])
        if data_dir[0] == '/':
          parent_dir = '/' + parent_dir
        df = pd.read_csv(os.path.join(parent_dir, 'maestro-v3.0.0.csv'))
        df['simple_midi_name'] = [midi_name[5:-5] for midi_name in df['midi_filename']]
        all_file_names = bf.listdir(data_dir)
        extracted_names = [extract_string(file_name) for file_name in all_file_names]
        classes = [find_composer(name, df) for name in extracted_names]

    dataset = ImageDataset(
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files(data_dir):
    dirs = bf.listdir(data_dir)
    return [data_dir + '/' + d for d in dirs]


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        arr = np.load(path)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict

