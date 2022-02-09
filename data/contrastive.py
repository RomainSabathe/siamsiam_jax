from typing import Callable

import jax
import jax.numpy as jnp
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")

# TODO: probably rename to "make_contrastive_dataset" or something.
def make_augmented_dataset(
    dataset_loading_fn: Callable, rng: jnp.array, *args, **kwargs
):
    data_order_rng, rng = jax.random.split(rng)
    augmentation_rng_1, augmentation_rng_2 = jax.random.split(rng)

    for key in [
        "rng",
        "data_order_rng",
        "augmentation_rng",
        "as_numpy",
    ]:
        if key in kwargs:
            kwargs.pop(key)
    kwargs["data_order_rng"] = data_order_rng
    kwargs["as_numpy"] = False

    dataset1 = dataset_loading_fn(augmentation_rng=augmentation_rng_1, *args, **kwargs)
    dataset2 = dataset_loading_fn(augmentation_rng=augmentation_rng_2, *args, **kwargs)

    def sanity_check(batch1, batch2):
        tf.assert_equal(
            tf.math.reduce_all(tf.equal(batch1["label"], batch2["label"])), True
        )
        return batch1, batch2

    return (
        tf.data.Dataset.zip((dataset1, dataset2)).map(sanity_check).as_numpy_iterator()
    )
