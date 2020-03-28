# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:46:17 2020

@author: Alain
"""

from train import get_unet
import h5py

"""
model = get_unet()

w = model.load_weights('weights.h5', by_name=True)

model.summary()
"""


def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
        

filename = "weights.h5"

with h5py.File(filename, 'r') as f:
    for dset in traverse_datasets(f):
        print('Path:', dset)
        print('Shape:', f[dset].shape)
        print('Data type:', f[dset].dtype)

