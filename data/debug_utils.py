from collections import defaultdict

from typing import Callable

import jax
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from tabulate import tabulate

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")

from .cifar10 import Cifar10WrappedBatch, load_cifar10_dataset


def dataset_sanity_check(rng: jnp.array, dataset_loading_fn, num_batches: int = -1):
    splits = ["train", "val", "query", "index", "test"]
    statistics = {split: defaultdict(lambda: 0) for split in splits}

    for split in tqdm(splits):
        ds = dataset_loading_fn(rng=rng, split=split, batch_size=1_024)
        for i, batch in enumerate(ds):
            if num_batches > 0 and i >= num_batches:
                break
            batch = Cifar10WrappedBatch(batch)
            for label in batch.y:
                label = int(label)
                statistics[split][label] += 1

    sizes = [
        (
            split,
            *[sub_dict[label] for label in range(10)],
            np.sum([v for v in sub_dict.values()]),
        )
        for (split, sub_dict) in statistics.items()
    ]
    print(tabulate(sizes, headers=[v for v in range(10)] + ["Total size"]))


def save_image_batch(img_batch: jnp.array, save_path: str):
    batch_size = img_batch.shape[0]
    size = np.sqrt(batch_size)
    if size != int(size):
        raise Exception(
            "This function does not support non-perfect square batch sizes."
        )
    size = int(size)

    imgs = rearrange(img_batch, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=size, b2=size)
    imgs = np.array(imgs)
    Image.fromarray(imgs).save(save_path)
    return


def compare_augmentations(
    dataset_loading_fn: Callable, save_path: str, split: str = "train", batch_size=8 * 8
):
    rng = jax.random.PRNGKey(0)

    size = np.sqrt(batch_size)
    if size != int(size):
        raise Exception(
            "This function does not support non-perfect square batch sizes."
        )
    size = int(size)
    common_kwargs = dict(rng=rng, split=split, shuffle=True, batch_size=batch_size)

    for i, batch in enumerate(
        dataset_loading_fn(apply_augmentations=False, **common_kwargs)
    ):
        batch = Cifar10WrappedBatch(batch)
        imgs1 = batch.x
        break

    for i, batch in enumerate(
        load_cifar10_dataset(apply_augmentations=True, **common_kwargs)
    ):
        batch = Cifar10WrappedBatch(batch)
        imgs2 = batch.x
        break

    imgs = np.stack([imgs1, imgs2], axis=0)
    imgs = rearrange(imgs, "n (b1 b2) h w c -> (b1 h) (b2 n w) c", b1=size, b2=size)
    imgs = np.array(imgs)
    Image.fromarray(imgs).save(save_path)


def dataset_checks(rng: jnp.array):
    dataset_sanity_check(rng, load_cifar10_dataset)
    compare_augmentations(
        load_cifar10_dataset, save_path="/mnt/c/Users/Romain/Desktop/img.png"
    )
