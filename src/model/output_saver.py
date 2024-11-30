import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

class OutputSaver(object):
    def __init__(self, save_path=None, random_seed=None):
        self.random_seed = random_seed

        self.save_path_numpy = '%s/%s/' % (save_path, 'numpy_files')
        os.makedirs(self.save_path_numpy, exist_ok=True)
        self.image_idx = 0

    def save(self, image_batch, recon_batch, label_true_batch, latent_batch):
        image_batch = image_batch.detach().numpy()
        recon_batch = recon_batch.detach().numpy()
        if label_true_batch is not None:
            label_true_batch = label_true_batch.detach().numpy()
        latent_batch = latent_batch.detach().numpy()
        image_batch = np.moveaxis(image_batch, 1, -1)
        recon_batch = np.moveaxis(recon_batch, 1, -1)
        if label_true_batch is not None:
            label_true_batch = np.moveaxis(label_true_batch, 1, -1)
        latent_batch = np.moveaxis(latent_batch, 1, -1)

        image_batch = squeeze_excessive_dimension(image_batch)
        recon_batch = squeeze_excessive_dimension(recon_batch)
        if label_true_batch is not None:
            label_true_batch = squeeze_excessive_dimension(label_true_batch)

        B, H, W, C = latent_batch.shape

        if label_true_batch is None:
            label_true_batch = np.empty((B,H,W))
            label_true_batch[:] = np.nan

        for image_idx in tqdm(range(B)):
            self.save_as_numpy(
                image=image_batch[image_idx, ...],
                recon=recon_batch[image_idx, ...],
                label=label_true_batch[image_idx, ...],
                latent=latent_batch[image_idx, ...].reshape((H * W, C)))
        return

    def save_as_numpy(self, image, recon, label, latent):
        with open(
                '%s/%s' %
            (self.save_path_numpy, 'sample_%s.npz' % str(self.image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f, image=image, recon=recon, label=label, latent=latent)
        self.image_idx += 1


def squeeze_excessive_dimension(batched_data):
    assert len(batched_data.shape) in [3, 4]
    if len(batched_data.shape) == 4 and batched_data.shape[-1] == 1:
        batched_data = batched_data.reshape(batched_data.shape[:3])
    return batched_data