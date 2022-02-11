import os
from typing import Generator, Optional

import jax
import jax.numpy as jnp
import tensorflow as tf
import imgaug.augmenters as iaa
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")

from .utils import Batch


class Cifar10WrappedBatch:
    def __init__(self, batch: Batch):
        self.batch = batch

    @property
    def x(self) -> jnp.array:
        return jnp.array(self.batch["image"])

    @property
    def y(self) -> jnp.array:
        return jnp.array(self.batch["label"])

    @property
    def y_one_hot(self):
        return jax.nn.one_hot(self.y, num_classes=10)


def load_cifar10_dataset(
    split: str,
    shuffle: bool,
    apply_augmentations: bool,
    batch_size: int,
    rng: Optional[jnp.array] = None,
    data_order_rng: Optional[jnp.array] = None,
    augmentation_rng: Optional[jnp.array] = None,
    as_numpy: bool = True,
    debug: bool = False,
) -> Generator[Cifar10WrappedBatch, None, None]:
    if rng is None and (data_order_rng is None and augmentation_rng is None):
        raise Exception(
            "Must provide either `rng` or `data_order_rng`+`augmentation_rng`."
        )
    if rng is not None:
        data_order_rng, augmentation_rng = jax.random.split(rng)

    split = {
        "val": "train[:1000]",
        "query": "train[1000:3000]",
        "index": "train[3000:5000]",
        "train": "train[5000:]",
        "test": "test",
    }[split]

    dataset = tfds.load("cifar10", split=split)
    if debug:
        # Only keeping classes 0 and 1 for faster training.
        dataset = dataset.filter(
            lambda sample: tf.math.logical_or(
                tf.math.equal(sample["label"], 0), tf.math.equal(sample["label"], 1)
            )
        )
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10, seed=int(data_order_rng[1]))

    dataset = dataset.batch(batch_size)

    if apply_augmentations:
        (
            aug_order_seed,
            crop_seed,
            flip_seed,
            sometimes8_seed,
            sometimes2_seed,
            brightness_seed,
            contrast_seed,
            saturation_seed,
            hue_seed,
            grayscale_seed,
            blur_seed,
        ) = jax.random.split(augmentation_rng, 11)

        _convert_seed = lambda jax_seed: int(jax_seed[1])
        sometimes8 = lambda aug: iaa.Sometimes(
            0.8, aug, seed=_convert_seed(sometimes8_seed)
        )
        sometimes2 = lambda aug: iaa.Sometimes(
            0.2, aug, seed=_convert_seed(sometimes2_seed)
        )

        augmentation_fn = iaa.Sequential(
            [
                iaa.Crop(
                    percent=(0.0, 0.2),
                    sample_independently=True,
                    keep_size=True,
                    seed=_convert_seed(crop_seed),
                ),
                iaa.Fliplr(0.5, seed=_convert_seed(flip_seed)),
                # Disable because imgaug has bug whereby the seed has no effect
                # ==> we can't reproduce the augmentations.
                # sometimes8(
                #   iaa.MultiplyBrightness(
                #       (0.6, 1.4), seed=_convert_seed(brightness_seed)
                #   ),
                # ),
                sometimes8(
                    iaa.LinearContrast((0.6, 1.4), seed=_convert_seed(contrast_seed))
                ),
                sometimes8(
                    iaa.MultiplySaturation(
                        (0.6, 1.4), seed=_convert_seed(saturation_seed)
                    )
                ),
                sometimes8(iaa.MultiplyHue((-0.1, 0.1), seed=_convert_seed(hue_seed))),
                sometimes2(iaa.Grayscale((0, 1), seed=_convert_seed(grayscale_seed))),
                # Not performed on Cifar experiments
                # sometimes2(
                #    iaa.GaussianBlur(sigma=(0.1, 0.5), seed=_convert_seed(blur_seed))
                # ),
            ],
            random_order=True,
            seed=_convert_seed(aug_order_seed),
        )

        def transform_batch_fn(batch: Batch):
            images = batch["image"]
            dtype = images.dtype
            shape = tf.shape(images)

            out = tf.numpy_function(
                func=lambda x: augmentation_fn(images=x), inp=[images], Tout=[dtype]
            )
            out = out[0]  # Only 1 output.
            out = tf.reshape(out, shape=shape)
            batch["image"] = out
            return batch

        dataset = dataset.map(transform_batch_fn)

    def remove_id(batch):
        batch.pop("id")
        return batch

    dataset = dataset.map(remove_id).prefetch(tf.data.AUTOTUNE)
    if as_numpy:
        dataset = tfds.as_numpy(dataset)
    return dataset
