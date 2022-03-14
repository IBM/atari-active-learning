# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
import os.path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class PixelDataSet(Dataset):
    """Saves pixel screens as torch tensors."""

    def __init__(self, dir = "./", dataset_size=(128, 128), normalize=False, limit=15000):
        """Constructor for pixel data sets."""
        os.makedirs(dir,exist_ok=True)
        self._dir = dir
        self.limit = limit

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.zeros((limit, 1, dataset_size[0], dataset_size[1]), device=self.device, dtype=torch.uint8)
        self._state_set = set()
        self.normalize = normalize
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        if self.normalization_valid:
            return
        data = self.data[:len(self)] / 255
        self.means = torch.mean(data, 0)
        self.stds = torch.std(data, 0, unbiased=False)
        # If all pixels have the same value, set std to be 1
        self.stds = torch.where(self.std < 1e-8, self.std, 1.0)
        self.normalized_data = (data - self.means) / self.stds
        self.normalization_valid = True

    def __len__(self):
        """Number of images in set."""
        return min(len(self._state_set),self.limit)

    def __getitem__(self, idx):
        """Gets the vector of the given index."""
        if self.normalize:
            # Recompute the normalization if data has been added since it was last computed.
            self.normalize_data()
            return self.normalized_data[idx]
        else:
            return self.data[idx] / 255


    def update_set(self, pixel_array):
        """Checks if a pixel array is in the data set already, and adds it if not. Does not cause disk access"""
        assert pixel_array.dtype == np.uint8
        assert isinstance(pixel_array, np.ndarray)
        b = pixel_array.data.tobytes()
        new = b not in self._state_set
        if not new:
            return False

        # Add the hash to the set. If it is rejected by reservoir sampling once, it will be rejected forever.
        self._state_set.add(b)

        # If the limit has already been reached, perform reservoir sampling to determine whether we keep the new pixel
        # array
        if len(self) >= self.limit:
            replace_idx = random.randrange(len(self)+1)
        else:
            replace_idx = len(self)

        # If not replacing the index, return False
        if replace_idx >= self.limit:
            return False

        # Save the tensor in the data array
        self.data[replace_idx] = torch.tensor(pixel_array, dtype=torch.uint8, device=self.device)[None, :, :]

        # If normalizing, set normalization to be out of date
        if self.normalize:
            self.normalization_valid = False

        return True

    def save(self):
        """Saves the images to the disk."""
        np.save(os.path.join(self._dir, 'screens.npy'), self.data.cpu().numpy()[:len(self)])

    def load(self):
        """load the images from the disk."""
        data = np.load(os.path.join(self._dir, 'screens.npy'))

        self._state_set.clear()
        for datum in data:
            self._state_set.add(datum.data.tobytes())

        self.data.fill_(0)
        self.data[:len(data)] = torch.tensor(data, dtype=torch.uint8, device=self.device)

    def delete(self):
        """Remove the dataset files"""
        def rm(p):
            if os.path.exists(p):
                os.remove(p)
        rm(os.path.join(self._dir,'screens.npy'))


def create_data_loaders(full_dataset, batch_size, train_distribution=0.95, **kwargs):
    """Creates train and test sets out of full_dataset."""
    train_size = int(train_distribution * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, train_set, test_set


def create_test_dataset(env, dir, n_rollouts, rollout_length):
    """Creates a dataset for testing purposes."""
    dataset = PixelDataSet(dir=dir)
    pixel_array = env.reset()
    dataset.update_set(pixel_array)
    action_space = env.action_space

    for n in range(n_rollouts):
        env.reset()
        for i in range(rollout_length):
            step_out = env.step(action_space.sample())
            dataset.update_set(step_out[0])
            # If done
            if step_out[2]:
                break

    dataset.save()

    return dataset
